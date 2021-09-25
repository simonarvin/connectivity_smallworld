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

**doi:** TBA

----

## Contents:

**Datasheets and simulation code:**
- [Generative small-world graphs + topological analysis](https://github.com/simonarvin/connectivity_smallworld/tree/main/small_world)
- [Kuramoto's coupled oscillators on the small-world graph](https://github.com/simonarvin/connectivity_smallworld/tree/main/kuramoto)

**Set-up:**
- [Installation](##installation)
- [Tests](##tests)
- [Requisites](##requisites)


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

Alternatively, see [the requisites list](Requisites).

## Tests

- *Reproduce small-world data:*

```python small_world/smallworld_simulation.py```

- *Reproduce small-world figures:*

```python small_world/smallworld_analysis.py```

- *Reproduce Kuramoto data:*

```python kuramoto/kuramoto_simulation.py```

- *Reproduce Kuramoto figures:*

```python kuramoto/kuramoto_analysis.py```

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
