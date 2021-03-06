Full datasets and simulation code
----
<p align="center">
  <img src="https://github.com/simonarvin/connectivity_smallworld/blob/main/misc/smallworld.svg" />
</p>

### Short- and long-range connections differentially modulate the small-world network’s dynamics and state

Simon Arvin<sup>1,2,†,\*</sup>, Andreas Nørgaard Glud<sup>1,†</sup> and Keisuke Yonehara<sup>2,\*</sup>

<sup>1</sup> Center of Experimental Neuroscience – CENSE, Department of Neurosurgery, Institute of Clinical Medicine, Aarhus University Hospital

<sup>2</sup> Danish Research Institute of Translational Neuroscience – DANDRITE, Nordic-EMBL Partnership for Molecular Medicine, Department of Biomedicine, Aarhus University, Ole Worms Allé 8, 8000 Aarhus C, Denmark

**Correspondence:**

Simon Arvin, sarv@dandrite.au.dk

Keisuke Yonehara, keisuke.yonehara@dandrite.au.dk

**doi:** [10.3389/fncom.2021.783474](10.3389/fncom.2021.783474)

----

## Contents:

**Datasheets and simulation code:**
- [Generative small-world graphs + topological analysis](https://github.com/simonarvin/connectivity_smallworld/tree/main/small_world)
- [Kuramoto's coupled oscillators on the small-world graph](https://github.com/simonarvin/connectivity_smallworld/tree/main/kuramoto)

**Set-up:**
- [Installation](#installation)
- [Tests](#tests)
- [Requisites](#requisites)
- [Authors](#authors)

## Installation

Download the datasets and simulation codes by cloning the repository:
```
git clone https://github.com/simonarvin/connectivity_smallworld.git
```

You may want to use a Conda or Python virtual environment to test this code, to avoid mixing up with your system dependencies.

Using pip and a virtual environment:

```python -m venv venv```

```source venv/bin/activate```

```(venv) pip install .```

> Remember to ```cd [path]``` to the root dataset directory.
> 
> [How to create a virtual environment in Windows](https://docs.python.org/3/library/venv.html).

Alternatively, see [the requisites list](#requisites).

## Tests

- *Reproduce small-world data:*

```python small_world/smallworld_simulation.py```

- *Reproduce small-world figures:*

```python small_world/smallworld_analysis.py```

- *Reproduce Kuramoto data:*

```python kuramoto/kuramoto_simulation.py```

- *Reproduce Kuramoto figures:*

```python kuramoto/kuramoto_analysis.py```

- *Reproduce Kuramoto stability/attraction data:*

```python kuramoto/kuramoto_simulation_stability.py```

- *Reproduce Kuramoto stability/attraction figures:*

```python kuramoto/kuramoto_analysis_stability.py```

- *Reproduce Kuramoto predictive power graph (**S1**):*

```python kuramoto/kuramoto_analysis_PPS.py```

----

## Requisites:
- Python 3.x (https://www.python.org/)
- networkx 2.6.x (https://networkx.org/)
- numpy 1.20.x (https://numpy.org/)
- scipy 1.5.0 (https://www.scipy.org/)
- pandas 1.3.x (https://pandas.pydata.org/)
- dominance-analysis 1.1.x (https://github.com/dominance-analysis/dominance-analysis)
- ppscore 1.2.0 (https://github.com/8080labs/ppscore/)

> e.g., ```pip install networkx==2.6.x```

## Authors:

- Simon Arvin, MD Hons candidate (sarv@dandrite.au.dk)
- Andreas Nørgaard Glud, MD PhD (angl@clin.au.dk)
- Keisuke Yonehara, DVM PhD (keisuke.yonehara@dandrite.au.dk)

----

<p align="center">
    <img src="https://github.com/simonarvin/eyeloop/blob/master/misc/imgs/aarhusuniversity.svg?raw=true" align="center" height="40">&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="https://github.com/simonarvin/eyeloop/blob/master/misc/imgs/dandrite.svg?raw=true" align="center" height="40">&nbsp;&nbsp;&nbsp;&nbsp;
    <img src="https://github.com/simonarvin/connectivity_smallworld/blob/main/misc/CENSE.jpg" align="center" height="40">
</p>
<p align="center">
    <a href="http://www.yoneharalab.com">
    <img src="https://github.com/simonarvin/eyeloop/blob/master/misc/imgs/yoneharalab.svg?raw=true" align="center" height="18">&nbsp;&nbsp;&nbsp;&nbsp;
    </a>
    </p>
