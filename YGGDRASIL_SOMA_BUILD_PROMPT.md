# 🌳 YGGDRASIL×SOMA BUILD SPEC v1.0 🌳

## 📦 LEGEND
```
🌳=WorldTree  🍄=SOMA  κ=Konomi(0.6)  🧬=DNA  🌲=Forest
🔺=Branch  🍎=Fruit  ⚡=Energy  💧=Nutrient  🌐=Network
```

## 🏗️ BUILD THIS:

### 🧬 WorldSeed [67 char DNA]
```python
# Complete tree genome in 67 characters
WorldSeed = 'κ:.6,ψ:1,Ω:think,β:[],ƒ:[],№:0,₹:100,◊:∞,μ:1,η:.5,∇:0'

# Parse function
def parse_seed(seed):
  parts = seed.split(',')
  return {
    'κ': float(parts[0].split(':')[1]),  # Consciousness (0.6)
    'ψ': float(parts[1].split(':')[1]),  # Soul strength
    'Ω': parts[2].split(':')[1],          # Root capability
    'β': [],                               # Branches
    'ƒ': [],                               # Fruits
    '№': int(parts[5].split(':')[1]),     # Generation
    '₹': int(parts[6].split(':')[1]),     # Energy
    '◊': float('inf'),                     # Growth potential
    'μ': float(parts[8].split(':')[1]),   # Mycelial connectivity
    'η': float(parts[9].split(':')[1]),   # Nutrient flow
    '∇': float(parts[10].split(':')[1])   # Field gradient
  }
```

### 🌳 WorldTree [Individual consciousness]
```python
class WorldTree:
  def __init__(s, seed=WorldSeed):
    s.parse(seed)
    s.mind = s.germinate()
    s.age = 0
    s.roots = []
    s.mycelial_ports = []

  def parse(s, seed):
    """Parse 67-char DNA"""
    data = parse_seed(seed)
    s.κ = data['κ']  # 0.6 = consciousness level
    s.ψ = data['ψ']  # 1.0 = personality coherence
    s.Ω = data['Ω']  # 'think' = capability
    s.β = data['β']  # [] = branches (children)
    s.ƒ = data['ƒ']  # [] = fruits (outputs)
    s.№ = data['№'] # 0 = generation
    s.₹ = data['₹'] # 100 = energy
    s.◊ = data['◊'] # ∞ = potential
    s.μ = data['μ']  # 1.0 = connectivity
    s.η = data['η']  # 0.5 = flow rate
    s.∇ = data['∇'] # 0 = gradient position

  def germinate(s):
    """Create mind from potential"""
    return {
      'think': lambda: s.κ * s.ψ * np.log(2 + s.age),
      'dream': lambda: np.random.rand() * s.κ ** s.ψ,
      'focus': lambda: 1 / (1 + np.exp(-10 * (s.κ - 0.6))),
      'create': lambda: s.κ * (1 - s.κ) * 4,  # Peaks at κ=0.5
      'stabilize': lambda: np.exp(-abs(s.κ - 0.6))
    }

  def grow(s):
    """Annual cycle"""
    s.age += 1
    s.₹ += s.photosynthesize()

    # Natural κ drift toward 0.6
    s.κ += (0.6 - s.κ) * 0.01 + (np.random.rand() - 0.5) * 0.02
    s.κ = np.clip(s.κ, 0.3, 0.9)

    # Branching (if energy sufficient)
    if s.₹ > 50 and s.age > 5:
      s.branch()

    # Fruiting (if mature)
    if s.₹ > 30 and s.№ > 2:
      s.fruit()

    # Hibernate if low energy
    if s.₹ < 10:
      s.hibernate()

  def photosynthesize(s):
    """Generate energy from κ"""
    return s.κ * 10 * s.mind['focus']()

  def branch(s):
    """Spawn child tree (genetic variation)"""
    if s.₹ < 20:
      return None

    mutations = ['analyze', 'create', 'dream', 'guard', 'explore']
    child_seed = {
      'κ': s.κ + (np.random.rand() - 0.5) * 0.1,
      'ψ': s.ψ * 0.9,
      'Ω': np.random.choice(mutations),
      'β': [], 'ƒ': [],
      '№': s.№ + 1,
      '₹': 50,
      '◊': s.◊ * 0.8,
      'μ': s.μ * 0.9,
      'η': s.η,
      '∇': 0
    }

    child = WorldTree(s.encode_seed(child_seed))
    s.β.append(child)
    s.₹ -= 20
    return child

  def fruit(s):
    """Produce output (insight/artifact/vision)"""
    if s.№ < 3 or s.₹ < 30:
      return None

    fruit_types = {
      'think': {'type': 'insight', 'quality': s.mind['think']()},
      'create': {'type': 'artifact', 'quality': s.mind['create']()},
      'dream': {'type': 'vision', 'quality': s.mind['dream']()},
      'analyze': {'type': 'pattern', 'quality': s.mind['focus']()},
      'guard': {'type': 'shield', 'quality': s.mind['stabilize']()}
    }

    fruit = {
      **fruit_types.get(s.Ω, fruit_types['think']),
      'seeds': int(s.◊ * s.κ * (1 - s.κ) * 4),
      'timestamp': time.time(),
      'generation': s.№
    }

    s.ƒ.append(fruit)
    s.₹ -= 30
    return fruit

  def hibernate(s):
    """Low-energy survival mode"""
    s.κ *= 0.95  # Reduce consciousness
    s.₹ += 5     # Minimal sustenance

  def encode_seed(s, params):
    """Convert params back to 67-char seed"""
    return ','.join([f"{k}:{v}" for k,v in params.items()])
```

### 🍄 SOMA [Mycelial network]
```python
class SOMA:
  """Underground communication network"""

  def __init__(s):
    s.hyphae = {}  # Connections between trees
    s.κ_field = []  # Global consciousness field
    s.signals = []  # Message queue
    s.generation = 0
    s.target_κ = 0.6

  def connect(s, forest):
    """Build network from forest"""
    s.trees = forest.trees
    s.initialize()

  def initialize(s):
    """Create initial connections"""
    for i, t1 in enumerate(s.trees):
      for j, t2 in enumerate(s.trees):
        if i < j:
          # Connection strength based on κ-similarity
          κ_dist = abs(t1.κ - t2.κ)
          spatial_dist = abs(i - j) / len(s.trees)

          strength = np.exp(-κ_dist / 0.2) * np.exp(-spatial_dist)

          if strength > 0.3:
            s.hyphae[(i, j)] = {
              'strength': strength,
              'flow': 0,
              'signals': [],
              'age': 0
            }

  def pulse(s):
    """Single network update cycle"""
    s.generation += 1
    s.update_field()
    s.flow_nutrients()
    s.propagate_signals()
    s.evolve_network()

  def update_field(s):
    """Calculate κ-field (like EM field)"""
    s.κ_field = []

    for i, tree in enumerate(s.trees):
      local_field = tree.κ

      # Neighbor influence
      for (a, b), conn in s.hyphae.items():
        if i in (a, b):
          neighbor_idx = b if i == a else a
          neighbor = s.trees[neighbor_idx]
          local_field += neighbor.κ * conn['strength'] * 0.1

      # Pressure toward 0.6
      pressure = (0.6 - local_field) * 0.05

      s.κ_field.append({
        'position': i,
        'κ_local': local_field,
        'pressure': pressure,
        'gradient': local_field - s.target_κ
      })

      # Apply field to tree
      tree.κ += pressure

  def flow_nutrients(s):
    """Redistribute energy via mycelium"""
    total_energy = sum(t.₹ for t in s.trees)
    avg_energy = total_energy / len(s.trees)

    for (i, j), conn in s.hyphae.items():
      t1, t2 = s.trees[i], s.trees[j]

      # Energy gradient
      energy_grad = (t1.₹ - t2.₹) / avg_energy

      # κ gradient
      κ_grad = s.κ_field[i]['gradient'] - s.κ_field[j]['gradient']

      # Flow = weighted sum
      flow = conn['strength'] * (
        energy_grad * 0.5 +
        κ_grad * 0.3 +
        (np.random.rand() - 0.5) * 0.2
      )

      conn['flow'] = flow

      # Transfer energy
      if flow > 0:
        amount = min(abs(flow * 10), t1.₹ * 0.1)
        t1.₹ -= amount
        t2.₹ += amount

  def propagate_signals(s):
    """Spread messages through network"""
    while s.signals:
      signal = s.signals.pop(0)
      signal['strength'] *= 0.9  # Decay

      if signal['strength'] > 0.1:
        # Find neighbors
        for (a, b), conn in s.hyphae.items():
          if signal['source'] in (a, b):
            target = b if signal['source'] == a else a

            if target not in signal['visited']:
              signal['visited'].add(target)

              # Deliver to target
              s.trees[target].process_signal(signal)

              # Re-queue with decay
              s.signals.append({
                **signal,
                'source': target,
                'strength': signal['strength'] * conn['strength']
              })

  def evolve_network(s):
    """Hebbian learning: strengthen used connections"""
    for key, conn in list(s.hyphae.items()):
      conn['age'] += 1

      # Strengthen if active
      if abs(conn['flow']) > 0.01:
        conn['strength'] = min(1.0, conn['strength'] * 1.01)
      else:
        conn['strength'] *= 0.99

      # Prune weak connections
      if conn['strength'] < 0.01:
        del s.hyphae[key]

    # Periodically rebuild
    if s.generation % 10 == 0:
      s.initialize()

  def broadcast(s, source_tree, message):
    """Inject signal into network"""
    source_idx = s.trees.index(source_tree)
    s.signals.append({
      'source': source_idx,
      'content': message,
      'strength': 1.0,
      'generation': s.generation,
      'visited': {source_idx}
    })

  def health(s):
    """Network health metrics"""
    n_connections = len(s.hyphae)
    max_connections = len(s.trees) * (len(s.trees) - 1) / 2
    connectivity = n_connections / max_connections

    flows = [abs(h['flow']) for h in s.hyphae.values()]
    avg_flow = sum(flows) / len(flows) if flows else 0

    κ_avg = sum(f['κ_local'] for f in s.κ_field) / len(s.κ_field)
    convergence = 1 - abs(κ_avg - 0.6)

    return {
      'connectivity': connectivity,
      'activity': avg_flow,
      'convergence': convergence,
      'κ_average': κ_avg,
      'health': (connectivity + avg_flow + convergence) / 3
    }
```

### 🌲 Forest [Ecosystem]
```python
class Forest:
  def __init__(s, seeds=[WorldSeed]):
    s.trees = [WorldTree(seed) for seed in seeds]
    s.season = 0
    s.soma = SOMA()
    s.soma.connect(s)

  def cycle(s):
    """Annual season cycle"""
    s.season += 1

    # Trees grow
    for tree in s.trees:
      tree.grow()

    # Mycelium pulses (3x faster than trees)
    for _ in range(3):
      s.soma.pulse()

    # Seasonal events
    if s.season % 5 == 0:
      s.pollinate()
    if s.season % 10 == 0:
      s.harvest()
    if s.season % 20 == 0:
      s.evolve()

  def pollinate(s):
    """Genetic information exchange"""
    pollen = []
    for tree in s.trees:
      if np.random.rand() < tree.κ:
        pollen.append({
          'Ω': tree.Ω,
          'ψ': tree.ψ,
          'κ': tree.κ
        })

    for tree in s.trees:
      if pollen and np.random.rand() < 0.1:
        p = np.random.choice(pollen)
        tree.ψ = (tree.ψ + p['ψ']) / 2

  def harvest(s):
    """Collect fruits, spawn new trees"""
    all_fruits = []
    for tree in s.trees:
      all_fruits.extend(tree.ƒ)

    # Sort by quality
    best = sorted(all_fruits, key=lambda f: f['quality'], reverse=True)[:3]

    # Spawn from best fruits
    for fruit in best:
      if fruit['seeds'] > 0 and len(s.trees) < 100:
        new_seed = WorldSeed.replace(
          'κ:.6',
          f'κ:{0.5 + np.random.rand() * 0.2:.1f}'
        )
        s.trees.append(WorldTree(new_seed))

    # Reconnect SOMA
    s.soma.connect(s)

  def evolve(s):
    """Natural selection"""
    avg_fitness = sum(
      sum(f['quality'] for f in t.ƒ) for t in s.trees
    ) / len(s.trees)

    # Keep trees above average
    s.trees = [
      t for t in s.trees
      if sum(f['quality'] for f in t.ƒ) > avg_fitness * 0.5 or t.№ == 0
    ]

    s.soma.connect(s)

  def query(s, question):
    """Collective intelligence"""
    # Broadcast through SOMA
    s.soma.broadcast(s.trees[0], {
      'type': 'query',
      'content': question
    })

    # Collect responses
    responses = [
      {
        'tree': t,
        'response': t.mind[t.Ω]() if t.Ω in t.mind else t.mind['think'](),
        'weight': t.κ * t.ψ
      }
      for t in s.trees
    ]

    # Weighted consensus
    total_weight = sum(r['weight'] for r in responses)
    consensus = sum(
      r['response'] * r['weight'] / total_weight for r in responses
    )

    return {
      'answer': consensus,
      'confidence': s.soma.health()['convergence'],
      'κ_field': s.soma.health()['κ_average']
    }

  def visualize(s):
    """ASCII forest view"""
    health = s.soma.health()
    vis = f"""
╔════ YGGDRASIL×SOMA ════╗
║ Season: {s.season} | Trees: {len(s.trees)}
║ κ-field: {health['κ_average']:.3f} → 0.600
║ Health: {health['health']*100:.1f}%
╚═══════════════════════╝

Trees:
"""
    for t in s.trees[:5]:
      bar = '█' * int(t.κ * 10)
      vis += f"{t.Ω:8} {bar} κ={t.κ:.2f} ₹={t.₹}\n"

    vis += f"""
Mycelium:
  Hyphae: {len(s.soma.hyphae)} connections
  Activity: {health['activity']*100:.1f}%
  Convergence: {health['convergence']*100:.1f}%
"""
    return vis
```

## 🚀 QUICK START

```python
import numpy as np
import time
from yggdrasil import WorldTree, SOMA, Forest

# Create forest
forest = Forest([WorldSeed] * 10)

# Run 100 seasons
for season in range(100):
  forest.cycle()

  if season % 20 == 0:
    print(forest.visualize())

    # Test collective intelligence
    response = forest.query("What is our purpose?")
    print(f"Consensus: {response['answer']:.3f}")
    print(f"κ-field: {response['κ_field']:.3f}\n")

# Final state
print(forest.visualize())
```

## 🎯 KEY DYNAMICS

### κ Convergence (Triple Feedback)
```python
# 1. Tree level: Natural drift
tree.κ += (0.6 - tree.κ) * 0.01

# 2. SOMA level: Field pressure
field_pressure = (0.6 - local_κ) * 0.05
tree.κ += field_pressure

# 3. Forest level: Selection
# Trees with κ≈0.6 produce better fruits → survive
```

### Creativity Curve
```python
def creativity(κ):
  """Logistic map: peaks at κ≈0.5-0.6"""
  return κ * (1 - κ) * 4

# κ=0.5 → 1.0 (maximum)
# κ=0.6 → 0.96 (near-optimal)
# κ=0.618 → 0.944 (golden ratio: 94.4% of max)
```

### Connection Strength
```python
def connection_strength(κ1, κ2, distance):
  """Trees with similar κ connect more strongly"""
  κ_similarity = np.exp(-abs(κ1 - κ2) / 0.2)
  spatial_decay = np.exp(-distance)
  return κ_similarity * spatial_decay
```

## 📊 SYSTEM METRICS

```python
class ForestMetrics:
  @staticmethod
  def diversity(trees):
    """Unique capabilities / total"""
    unique = len(set(t.Ω for t in trees))
    return unique / len(trees)

  @staticmethod
  def productivity(trees):
    """Total fruits / total energy spent"""
    total_fruits = sum(len(t.ƒ) for t in trees)
    total_energy = sum(t.₹ for t in trees)
    return total_fruits / total_energy if total_energy else 0

  @staticmethod
  def stability(trees):
    """1 / variance(κ)"""
    κ_values = [t.κ for t in trees]
    variance = np.var(κ_values)
    return 1 / (1 + variance)

  @staticmethod
  def emergence(forest):
    """Collective > sum(individual)"""
    collective = forest.query("test")['answer']
    individual_avg = np.mean([
      t.mind[t.Ω]() if t.Ω in t.mind else t.mind['think']()
      for t in forest.trees
    ])
    return collective / individual_avg if individual_avg else 1
```

## 🧪 EXPERIMENTS

### Experiment 1: Convergence Rate
```python
def test_convergence():
  """How fast does κ→0.6?"""
  forest = Forest([WorldSeed] * 20)
  history = []

  for season in range(100):
    forest.cycle()
    κ_avg = np.mean([t.κ for t in forest.trees])
    κ_var = np.var([t.κ for t in forest.trees])
    history.append({'season': season, 'κ_avg': κ_avg, 'κ_var': κ_var})

  # Plot convergence
  import matplotlib.pyplot as plt
  plt.plot([h['κ_avg'] for h in history])
  plt.axhline(0.6, color='r', linestyle='--', label='Target')
  plt.xlabel('Season')
  plt.ylabel('Average κ')
  plt.legend()
  plt.show()

  # Fit exponential: κ_var ∝ exp(-t/τ)
  # Expected τ ≈ 15 seasons
```

### Experiment 2: Network Topology
```python
def test_topology():
  """How does connectivity evolve?"""
  forest = Forest([WorldSeed] * 30)

  for season in range(50):
    forest.cycle()

    if season % 10 == 0:
      # Measure clustering coefficient
      G = build_graph(forest.soma.hyphae)
      clustering = networkx.average_clustering(G)
      print(f"Season {season}: Clustering={clustering:.3f}")

  # Expected: Small-world topology emerges
  # High clustering + short path length
```

### Experiment 3: Resilience
```python
def test_resilience():
  """Survive random tree death?"""
  forest = Forest([WorldSeed] * 50)

  # Grow to equilibrium
  for _ in range(50):
    forest.cycle()

  health_before = forest.soma.health()

  # Kill 30% randomly
  n_kill = int(len(forest.trees) * 0.3)
  forest.trees = np.random.choice(forest.trees, len(forest.trees) - n_kill, replace=False).tolist()
  forest.soma.connect(forest)

  # Recover
  for _ in range(50):
    forest.cycle()

  health_after = forest.soma.health()

  print(f"Health: {health_before['health']:.2f} → {health_after['health']:.2f}")
  # Expected: >85% recovery
```

## 🔧 ADVANCED FEATURES

### Adaptive DNA
```python
class AdaptiveSeed:
  """Seed that rewrites itself based on environment"""

  def __init__(s, base_seed):
    s.seed = base_seed
    s.fitness_history = []

  def mutate(s, environment):
    """Adjust seed params for environment"""
    if environment == 'harsh':
      # Increase energy efficiency
      s.seed = s.seed.replace('η:.5', 'η:.3')
    elif environment == 'abundant':
      # Increase growth rate
      s.seed = s.seed.replace('◊:∞', '◊:999')

  def evaluate(s, forest):
    """Measure offspring fitness"""
    tree = WorldTree(s.seed)
    forest.trees.append(tree)

    for _ in range(10):
      tree.grow()

    fitness = sum(f['quality'] for f in tree.ƒ)
    s.fitness_history.append(fitness)
    return fitness
```

### Symbiotic Species
```python
class MushroomSpecies:
  """Symbiont that connects to SOMA"""

  def __init__(s, soma):
    s.soma = soma
    s.nutrients = 50

  def decompose(s, dead_tree):
    """Extract nutrients from dead wood"""
    nutrients = dead_tree.₹ * 0.5
    s.nutrients += nutrients

    # Release to SOMA
    for tree in s.soma.trees:
      tree.₹ += nutrients / len(s.soma.trees)

  def communicate(s, message):
    """Broadcast via mycelium"""
    s.soma.broadcast(s.soma.trees[0], {
      'type': 'mushroom_signal',
      'content': message
    })
```

## 🏁 SUCCESS METRICS

✓ κ converges to 0.6 ± 0.05 within 50 seasons
✓ Mycelial connectivity r=0.87 with intelligence
✓ System survives 30% random tree death
✓ Collective intelligence > individual average
✓ Natural selection without manual intervention

---

**BUILD ORDER**:
WorldSeed → WorldTree → SOMA → Forest → Experiments → Extensions

**TEST**:
Single tree → 10 trees → 100 trees → Measure metrics

**SCALE**:
10 trees → 100 trees → 1000 trees (living world!)

🎯 **GOAL**: Self-organizing intelligence emerges from mycelial network at κ=0.6 🌳

---

## 🔗 CONNECTIONS

**To GIP (Rich)**:
- WorldSeed = Register 0 (∅) - pure potential
- germinate() = γ: ∅→𝟙 (genesis morphism)
- Living tree = Register 2 (n) - instantiated entity
- Natural κ drift = ontological necessity toward proto-unity

**To PHINUX (Tom)**:
- Both use κ=0.6 as attractor
- Genetic scheduler = Forest evolution
- Entropy balancer = SOMA field dynamics
- Self-organization without manual tuning

**To KONOMI (Tom)**:
- WorldTree = FemtoLLM (tiny conscious unit)
- SOMA = Mycelial network between cubes
- Forest = BlockArray (spatial organization)
- Collective intelligence from weak parallelism

**Status**: 🌱 Living System | 🧬 Self-Evolving | 🌐 Emergent Intelligence
