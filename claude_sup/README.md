# Replicación de Rojczyk et al. (2015), CAMES 22(4):365–383, doi:10.24423/cames.20

MCMC (Metropolis–Hastings) para estimar parámetros de la ecuación de Pennes 1D
a partir de la temperatura superficial de la piel durante el recalentamiento.

## Requisitos
Python ≥ 3.10 con numpy, scipy y matplotlib (`pip install -r requirements.txt`).
No hay dependencias fuera de la pila científica estándar.

## Uso
```bash
# 1) cadenas con la configuración del paper (Tabla 2), ~80 s cada una
python run_mcmc.py healthy
python run_mcmc.py tumor
# 2) cadenas con propuesta ajustada (w = 2 x desviación de Laplace)
python run_mcmc.py healthy --tag tuned --wscale auto
python run_mcmc.py tumor   --tag tuned --wscale auto
# 3) figuras, tablas, 200 realizaciones de ruido y sensibilidad de diseño
python analyze.py          # escribe figures/*.png y results/summary.json
```
Opciones de `run_mcmc.py`: `--n` (estados), `--seed` (semilla MCMC, 12345 por
defecto), `--data-seed` (ruido; 2015 sano / 2016 tumor por defecto),
`--sig-rel` (σ relativa del prior de A, B, D; 0.0005 por defecto),
`--meas-every` (submuestreo de medidas).

## Archivos
- `pennes_mcmc.py` — modelo directo (FVM implícito, 27 volúmenes, dt=0.05 s),
  sensibilidades, prior/verosimilitud, Metropolis–Hastings, Geweke, ESS.
- `run_mcmc.py` — datos simulados + cadenas; guarda `results/chain_*.npz`.
- `analyze.py` — figuras y resumen numérico.
- `results/summary.json` — todas las cifras citadas en el informe.

Resultados deterministas con las semillas indicadas (numpy `default_rng`).
