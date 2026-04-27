# Metrología de Defectos en Piezas de Forja mediante SG + Gaussian Splatting

> **Objetivo:** Caracterizar geométricamente defectos superficiales (bollos, picos, abolladuras) en piezas de revolución sin depender de un modelo CAD, con alta tolerancia al ruido láser y en ciclos de producción < 2 segundos.
>
> **Magnitudes de salida:** Volumen del material desplazado, semiejes principales (largo y ancho), profundidad máxima, localización sub-píxel del centro.

---

## 1. Fundamentos: Por qué SG + Gaussian Splatting

### 1.1 El problema de la ambigüedad geométrica

En inspección sin CAD, el reto central es separar la geometría nominal de la pieza del defecto real. En una superficie de revolución (cilindro), un golpe suficientemente abierto puede parecer, localmente, un plano perfecto. Sin referencia, el sistema no puede discernir si la pieza "es plana" o "es un cilindro golpeado".

### 1.2 Limitaciones de la metrología clásica sobre nubes de puntos

| Característica | Nubes de puntos (tradicional) | SG + Gaussian Splatting (propuesto) |
|---|---|---|
| Sensibilidad al ruido | Alta — error directo en Z | Baja — promediado estadístico |
| Resolución espacial | Discreta (paso del láser) | Sub-píxel, continua/analítica |
| Precisión en aristas | Pobre — discontinuidades | Alta — interpolación C² |
| Velocidad | O(N log N) para mallas | O(K) funcional (K << N) |
| Geometría compleja | Falla en aristas y curvas | Reconstruye la "intención de diseño" |

### 1.3 Dualidad espacio-frecuencia

Mientras el filtro SG opera en el dominio espacial (ajuste polinómico local), los splats gaussianos actúan como funciones de base que capturan la energía del defecto en el dominio frecuencial. Esta dualidad permite una caracterización invariante a la rotación de la pieza.

---

## 2. Sistema de Coordenadas

El sistema opera en **Coordenadas del Mundo**:

| Eje | Representación física |
|---|---|
| Z | Altura real de la pieza respecto a la base/cinta |
| X | Dirección transversal al movimiento |
| Y | Dirección longitudinal al movimiento |

**Convención:** Un golpe hacia adentro produce una disminución local de Z (valle geométrico). La metrología diferencial es siempre `Z_fantasma − Z_medida > 0` para defectos de tipo bollo.

---

## 3. Pilar I — Filtro Savitzky-Golay como Estimador de Curvatura

El filtro SG no es solo un suavizador: es un **proyector polinómico local** que estima derivadas de la superficie con mínima distorsión espectral.

### 3.1 Modelo cuadrático local (2D)

Para cada ventana de análisis $w \times w$, se ajusta un paraboloide centrado en el píxel $(x_0, y_0)$:

$$f(x,y) = a(x-x_0)^2 + b(y-y_0)^2 + c(x-x_0)(y-y_0) + d(x-x_0) + e(y-y_0) + f_0$$

Los seis coeficientes $(a, b, c, d, e, f_0)$ se extraen en paralelo para toda la imagen mediante convoluciones separables (implementación con OpenCV).

### 3.2 Información geométrica extraída

- **$f_0$ (offset):** Altura promediada localmente.
- **$(d, e)$ (gradiente $\nabla f$):** Inclinación de la superficie; indica la orientación del defecto.
- **Hessiano $\mathbf{H}$:** Define la curvatura local:

$$\mathbf{H} = \begin{pmatrix} 2a & c \\ c & 2b \end{pmatrix}$$

- **Autovalores de $\mathbf{H}$ ($\kappa_1, \kappa_2$):** Curvaturas principales. Firma topológica de la geometría nominal:
  - Plano: $\kappa_1, \kappa_2 \approx 0$
  - Cilindro: $\kappa_1 \approx 1/R$, $\kappa_2 \approx 0$
  - Arista: bimodalidad en las normales del vecindario

### 3.3 Localización sub-píxel del vértice

El mínimo del paraboloide se obtiene resolviendo $\nabla f = 0$:

$$\begin{pmatrix} \Delta x \\ \Delta y \end{pmatrix} = -\mathbf{H}^{-1} \begin{pmatrix} d \\ e \end{pmatrix}$$

Se puede refinar iterativamente con Newton + backtracking line-search (ya implementado en `sg_2D.ipynb`).

### 3.4 Estimación de parámetros gaussianos desde la parábola

En 1D: dada la equivalencia local $-A \exp(-x^2 / 2\sigma^2) \approx -A + A x^2/(2\sigma^2)$, se obtiene:

$$A = -f_v, \quad \sigma = \sqrt{\frac{A}{2|a|}}$$

En 2D, los semiejes del elipsoide del defecto son $\sigma_{1,2} = \sqrt{A / (2|\kappa_{1,2}|)}$.

---

## 4. Pilar II — Gaussian Splatting como Modelo Continuo del Defecto

En lugar de tratar el defecto como una resta discreta de puntos, se representa como una **Mezcla de Gaussianas (GMM)**. Cada "splat" es:

$$G_k(x, y) = A_k \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)\right)$$

con parámetros: posición $\boldsymbol{\mu}_k$, forma/escala $\boldsymbol{\Sigma}_k$ y amplitud $A_k$.

### 4.1 Inicialización desde SG (Warm Start)

Para cumplir el tiempo de ciclo, los splats se inicializan directamente con los resultados del filtro SG:

| Parámetro del splat | Inicialización desde SG |
|---|---|
| Centro $\boldsymbol{\mu}$ | Vértice del paraboloide: $\boldsymbol{\mu} = \mathbf{x}_0 - \mathbf{H}^{-1}\nabla f_0$ |
| Covarianza $\boldsymbol{\Sigma}$ | Curvatura invertida: $\boldsymbol{\Sigma} = -A \mathbf{H}^{-1}$ |
| Amplitud $A$ | Profundidad en el vértice: $A = -f_v$ |

### 4.2 Refinamiento en tiempo real

Si el residuo del ajuste supera un umbral (defecto complejo o asimétrico), el splat "maestro" se divide en splats "satélites" que se ajustan con Levenberg-Marquardt, limitado a < 15 iteraciones para garantizar el tiempo de ciclo.

### 4.3 Ventaja metrológica clave

El volumen del defecto es una **integral analítica cerrada**:

$$V = \sum_k A_k \cdot 2\pi \sqrt{\det(\boldsymbol{\Sigma}_k)}$$

Esto elimina el error de cuantización y la dependencia del paso de muestreo del láser.

---

## 5. Pilar III — Reconstrucción de la Superficie Fantasma (Inpainting Geométrico)

Para medir un volumen necesitamos la referencia de "cómo debería ser la pieza sana" en la zona del defecto. Al no usar CAD, se aplica inpainting geométrico.

### 5.1 Clasificación topológica automática del vecindario sano

Se analiza la corona exterior al defecto (anillo sano) mediante los autovalores del Hessiano:

- **Plano:** $\kappa_1, \kappa_2 \approx 0$ → extrapolación plana.
- **Cilindro:** $\kappa_1 \approx 1/R$, $\kappa_2 \approx 0$ → extrapolación cilíndrica con radio $R$ estimado.
- **Arista (90°):** bimodalidad en las normales → se detectan dos geometrías contiguas.

### 5.2 Extrapolación competitiva en aristas

Si el golpe está en la arista entre una base plana y una pared cilíndrica, ambas geometrías se extrapolaran hacia el centro del defecto. La **Superficie Fantasma** es la intersección analítica de las dos proyecciones. Esta es la situación más crítica para los defectos tipo "bollo" en piezas de forja.

---

## 6. Metrología Final

Con el modelo de splats optimizado y la superficie fantasma reconstruida:

| Magnitud | Fórmula |
|---|---|
| **Volumen** | $V = \sum_k A_k \cdot 2\pi\sqrt{\det(\boldsymbol{\Sigma}_k)}$ |
| **Profundidad máxima** | $d = \max\bigl(Z_{\text{fantasma}} - \sum_k G_k\bigr)$ |
| **Largo / Ancho** | Semiejes del OBB: raíces cuadradas de los autovalores de la covarianza global de los splats |
| **Centro del defecto** | $\boldsymbol{\mu}_{\text{global}} = \sum_k w_k \boldsymbol{\mu}_k$ (media ponderada por $A_k$) |

---

## 7. Pipeline Completo

```
Nube de puntos (sensor láser)
        │
        ▼
[Transformación al sistema del mundo]
        │
        ▼
[Filtro SG 2D] ──► Coeficientes (a,b,c,d,e,f₀) por píxel
        │                   │
        │          Autovalores de H → clasificación topológica
        │                   │
        │          Vértice sub-píxel → inicialización splats
        │
        ▼
[Segmentación del defecto]  (máscara basada en curvatura)
        │
        ├──► [Anillo sano] ──► Reconstrucción Superficie Fantasma
        │
        └──► [Zona defecto] ──► Ajuste GMM (Warm Start + LM refinement)
                                        │
                                        ▼
                              [Metrología analítica]
                              Volumen / Profundidad / Largo / Ancho
```

---

## 8. Experimentos Planificados

### Experimento 0 — Validación sobre datos 1D simulados *(completado en `sg.ipynb`, `sg_defectos_Simulados.ipynb`)*

Verificación de la extracción de $(A, \sigma)$ desde coeficientes SG con datos sintéticos. Base de la regresión de calibración.

### Experimento 1 — Validación 2D en superficies simuladas *(completado en `sg_2D.ipynb`)*

Análisis de sensibilidad al tamaño de ventana (5–111 px). Extracción de direcciones principales y correlación con ground truth.

### Experimento 2 — Aplicación a defectos reales con SG *(completado en `sg_defectos_reales.ipynb`, `sg_2D_defectos_reales.ipynb`)*

Pipeline SG sobre escaneos `.raw` reales. Calibración por regresión para convertir coeficientes a magnitudes físicas.

### Experimento 3 — Inicialización de Splats desde SG *(pendiente)*

Conectar la salida del filtro SG (vértice, $\mathbf{H}$, $A$) con la parametrización de un splat gaussiano. Validar que `Warm Start` reproduce fielmente el defecto sin optimización adicional.

### Experimento 4 — Refinamiento GMM sobre datos reales *(pendiente)*

Ajuste Levenberg-Marquardt de splats sobre los defectos reales (bollos y picos). Comparar volumen medido con ground truth de la base de datos existente.

### Experimento 5 — Reconstrucción de Superficie Fantasma en aristas *(pendiente)*

Clasificación topológica automática del vecindario sano. Validar en las piezas con defectos en aristas (zona base-pared).

### Experimento 6 — Validación metrológica completa *(pendiente)*

Pipeline end-to-end sobre el dataset `defectos_reales/bollos/` y `defectos_reales/picos/`. Métricas: error en volumen, largo, ancho y profundidad respecto a mediciones de referencia.



Para estimar la forma original de la pieza (la "superficie fantasma") basándote únicamente en el entorno del defecto, el sistema debe realizar una clasificación topológica basada en geometría diferencial. No necesitas saber de antemano qué pieza es; la respuesta está en los puntos sanos que rodean la máscara del golpe.

Aquí tienes el resumen del procedimiento para deducir la forma:
1. El Anillo de Sondeo (Sensing Ring)

Una vez detectado y segmentado el defecto, el algoritmo selecciona una "corona" de puntos (vecindario sano) inmediatamente exterior al borde del golpe. En esta zona, aplicas el filtro de Savitzky-Golay (SG) para extraer las propiedades locales de cada punto.
2. Clasificación por Invariantes Geométricos

Para cada splat o punto en ese anillo, analizamos los autovalores (κ1​,κ2​) de la matriz Hessiana proporcionada por el filtro SG. Esto nos permite clasificar el entorno en tres escenarios posibles:

    Escenario A: Plano Puro

        Condición: κ1​≈0 y κ2​≈0.

        Interpretación: Si todo el anillo tiene curvatura nula y las normales son paralelas, el golpe está en una cara plana (como la base del cilindro).

    Escenario B: Superficie de Revolución (Cilindro)

        Condición: κ1​≈1/R (constante) y κ2​≈0.

        Interpretación: Las normales convergen hacia un eje central. El sistema deduce que el entorno es una pared lateral curva.

    Escenario C: Arista o Esquina (Bimodalidad)

        Condición: El anillo contiene dos grupos de puntos con normales perpendiculares entre sí (unos con κ≈0 y otros con κ≈1/R).

        Interpretación: El defecto ha "borrado" la arista que une la base con la pared.

3. Reconstrucción de la Superficie Fantasma

Una vez identificada la topografía, el sistema aplica una extrapolación competitiva:

    Ajuste de Primitivas: Si el entorno es bimodal (arista), el algoritmo ajusta por mínimos cuadrados un plano ideal a los puntos de la base y un cilindro ideal a los puntos de la pared lateral.

    Intersección Analítica: La superficie fantasma se define como la unión de estas dos funciones. La arista teórica es el lugar geométrico donde ambas funciones se encuentran.

    Inpainting de Curvatura: Para superficies más complejas (como splines), se utiliza la propagación de la curvatura desde el borde hacia el interior, minimizando la energía de flexión (bending energy) para que la transición sea suave.

4. Determinación del "Cero Metrológico"

La superficie fantasma estimada se convierte en tu valor de referencia (Z-nominal).

    Cualquier desviación medida por los splats del golpe respecto a esta superficie reconstruida se contabiliza como volumen de material faltante.

    Esto permite que, incluso si la pieza entra ligeramente inclinada en la cinta transportadora, el "cero" se mueva con la pieza, eliminando errores de alineación.

Resumen Lógico para Copilot:

Para programar esto, el flujo es:

    Extraer normales y curvaturas del anillo exterior.

    Agrupar (clustering) normales similares.

    Si hay un grupo: Extrapolar superficie única.

    Si hay dos grupos: Calcular intersección de superficies (Arista).

    Restar modelo - realidad.