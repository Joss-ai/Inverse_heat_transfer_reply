# Ecuación de Pennes: MDF implícitas frente a MVF

Desarrollo matemático autocontenido (LaTeX y PDF) y simulación reproducible en Python.

## Archivos
- `Desarrollo_Pennes_MDF.tex`: fuente LaTeX (se puede abrir en Overleaf).
- `Desarrollo_Pennes_MDF.pdf`: documento maquetado.
- `comparar_mdf_mvf.py`: resolución independiente por diferencias finitas con Robin de orden 1 y orden 2, y por volúmenes finitos.
- `comparacion_mdf_mvf.csv`: resultados a tiempos elegidos para N=27,54,108.
- `comparacion_sano.png` y `comparacion_tumoral.png`: gráficos generados.

## Instalar y ejecutar

Se requiere Python 3.10+, `numpy` y `matplotlib`:

```bash
python -m pip install numpy matplotlib
python comparar_mdf_mvf.py
```

Para recompilar el documento:

```bash
pdflatex Desarrollo_Pennes_MDF.tex
pdflatex Desarrollo_Pennes_MDF.tex
```

El código mantiene delta_t = 0.05 s, y modela 15 s de enfriamiento y 50 s de recalentamiento. Se usa N=27 intervalos MDF, frente a N=27 volúmenes MVF, por lo que las incógnitas de ambos métodos no coinciden espacialmente. Los parámetros físicos derivan de la tabla 1 del artículo de Rojczyk et al. (2015). Esta es una implementación independiente, no el MATLAB original.
