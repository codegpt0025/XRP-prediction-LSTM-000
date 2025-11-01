# Jules: Build Elite Adaptive Prediction System

## What You're Building
A world-class prediction system using cutting-edge techniques from top hedge funds and AI labs. This is NOT a toy project - it must be competitive with Renaissance Technologies, Citadel, and Two Sigma.

**Read ARCHITECTURE.md first.** It contains the complete design.

## Core Principle
**ONE INTELLIGENT BRAIN, NOT A COMMITTEE OF WEAK MODELS.**

You're building a single meta-learned adaptive network that:
- Learns how to learn (meta-learning)
- Never stops improving (online learning)
- Understands causality (not just correlation)
- Evolves its own architecture (neural architecture search)
- Makes intelligent decisions (reinforcement learning)
- Remembers like humans (episodic memory)

## Implementation Roadmap

### Phase 1: Core Brain (src/core/)

#### 1.1 Adaptive Encoder (`encoder.py`)
```python
class AdaptiveEncoder(nn.Module):
    """Learns optimal feature representation"""
    
    def __init__(self, input_dim, hidden_dim=512):
        # Multi-head self-attention over raw inputs
        # Transformer encoder with 6 layers
        # Layer normalization and residual connections
        # Adaptive pooling based on importance
        
    def forward(self, x):
        # Returns: learned representation
        # No manual feature engineering
        # Automatic feature discovery
```

**Key techniques:**
- Self-attention mechanisms (like GPT)
- Learned positional encoding
- Cross-modal attention (if multiple data types)
- Dynamic feature selection

#### 1.2 Causal Attention (`causal_attention.py`)
```python
class CausalAttention(nn.Module):
    """Attention that understands causality"""
    
    def __init__(self):
        # Implements Granger causality in attention
        # Masked attention (future can't cause past)
        # Learnable causal graph
        
    def forward(self, x):
        # Returns: causally-weighted representations
        # Identifies: what causes what
```

**Implementation:**
- Use PC algorithm or neural causal discovery
- Integrate with DoWhy library
- Learn directed acyclic graph (DAG)
- Attention weights respect causal structure

#### 1.3 Memory Network (`memory.py`)
```python
class MemoryAugmentedNetwork(nn.Module):
    """External memory for market patterns"""
    
    def __init__(self, memory_slots=256, slot_dim=512):
        self.memory = nn.Parameter(torch.randn(memory_slots, slot_dim))
        self.fast_weights = None  # Recent patterns
        self.slow_weights = None  # Long-term structure
        
    def read(self, query):
        # Content-based addressing
        # Returns similar past situations
        
    def write(self, value, gate):
        # Update memory with new patterns
        # Decay old, irrelevant memories
```

**Based on:** Neural Turing Machines, Differentiable Neural Computer

#### 1.4 RL Decision Layer (`rl_agent.py`)
```python
class RLDecisionAgent:
    """Learns optimal prediction strategy"""
    
    def __init__(self):
        # PPO or SAC algorithm
        # State: market features + uncertainty + memory
        # Actions: {predict_high, predict_low, wait, uncertain}
        # Reward: Sharpe ratio, directional accuracy
        
    def select_action(self, state):
        # Returns: action + confidence
        # Learns when to be bold vs cautious
```

**Use:** Stable-Baselines3 with custom environment

#### 1.5 Meta-Learner (`meta_learner.py`)
```python
class MAMLMetaLearner:
    """Model-Agnostic Meta-Learning"""
    
    def __init__(self, model):
        self.model = model
        self.meta_optimizer = Adam(lr=0.001)
        
    def meta_train(self, task_distribution):
        # Inner loop: adapt to specific task
        # Outer loop: update initialization
        # Goal: find params that adapt quickly
        
    def fast_adapt(self, new_data, steps=5):
        # Adapts to new regime in <10 examples
        # Uses learned initialization
```

**Critical:** This is what makes system intelligent - learns to learn

#### 1.6 Online Learner (`online_learner.py`)
```python
class OnlineLearner:
    """Continuous learning without forgetting"""
    
    def __init__(self, model):
        self.model = model
        self.importance_weights = {}  # EWC
        
    def update(self, new_batch):
        # Streaming gradient descent
        # Protect important parameters (EWC)
        # No full retraining needed
        
    def consolidate(self):
        # Compute Fisher information
        # Identify important parameters
```

**Prevents:** Catastrophic forgetting

#### 1.7 Neural Architecture Search (`nas.py`)
```python
class EfficientNAS:
    """Evolves network architecture"""
    
    def __init__(self):
        # Search space: layers, units, connections, activations
        # Use ENAS or DARTS
        
    def search(self, validation_data, budget=100):
        # Discovers optimal architecture
        # Runs overnight weekly
        # Returns: best architecture found
        
    def evolve_architecture(self, current_arch):
        # Gradual evolution
        # Tests variations
        # Keeps what works
```

### Phase 2: Data Intelligence (src/data/)

#### 2.1 Unified Pipeline (`unified_pipeline.py`)
```python
class DataIntelligence:
    """Single intelligent data system"""
    
    def __init__(self):
        self.sources = self.discover_sources()
        self.quality_scores = {}
        
    def discover_sources(self):
        # Tries multiple APIs
        # Scores by reliability, latency, cost
        # Returns: ranked list
        
    def collect(self, asset, lookback='adaptive'):
        # Multi-modal: prices, sentiment, macro
        # Real-time cleaning
        # Anomaly detection
        # Returns: clean tensor
        
    def adapt_to_new_asset(self, asset):
        # Zero-shot: works on new symbols
        # Transfer learned patterns
```

**No separate preprocessing!** Clean as you collect.

#### 2.2 Smart Collectors (`collectors.py`)
```python
class AdaptiveCollector:
    """Learns API behavior"""
    
    def __init__(self, api_name):
        self.rate_limits = self.learn_limits()
        self.quality_score = 0.5
        
    def fetch(self, query):
        # Adaptive rate limiting
        # Automatic retries
        # Quality scoring
        # Caching
        
    def learn_limits(self):
        # Discovers actual rate limits
        # Not hardcoded
```

### Phase 3: Experience System (src/experience/)

#### 3.1 Episodic Memory (`episodic.py`)
```python
class EpisodicMemory:
    """Human-like memory system"""
    
    def __init__(self, capacity=100000):
        self.episodes = []
        self.embeddings = None
        
    def remember(self, state, action, outcome, surprise):
        # Emotional tagging (big moves remembered better)
        # Semantic clustering
        # Temporal decay
        episode = {
            'state': state,
            'action': action,
            'outcome': outcome,
            'surprise': surprise,
            'timestamp': now(),
            'importance': self.compute_importance(surprise, outcome)
        }
        self.episodes.append(episode)
        
    def recall(self, current_state, k=10):
        # Content-based retrieval
        # Returns: most similar past experiences
        similarities = cosine_similarity(current_state, self.embeddings)
        return top_k(self.episodes, similarities, k)
```

#### 3.2 Counterfactual Reasoning (`counterfactual.py`)
```python
class CounterfactualReasoner:
    """What if I had decided differently?"""
    
    def analyze_decision(self, past_decision):
        # Simulate alternative actions
        # Estimate causal effects
        # Learn from hypotheticals
        
    def identify_biases(self):
        # Systematic errors
        # When does system fail?
        # Returns: failure patterns
```

#### 3.3 Regime Detector (`regime.py`)
```python
class RegimeDetector:
    """Hidden Markov Model for markets"""
    
    def __init__(self, n_regimes=5):
        # States: trending, mean-reverting, volatile, quiet, crisis
        self.hmm = GaussianHMM(n_components=n_regimes)
        
    def detect(self, market_data):
        # Returns: current regime + confidence
        
    def get_regime_strategy(self, regime):
        # Different meta-policy per regime
        # Learned, not hardcoded
```

### Phase 4: Self-Improvement (src/improvement/)

#### 4.1 Performance Analyzer (`analyzer.py`)
```python
class SelfImprovement:
    """Analyzes and improves itself"""
    
    def analyze_errors(self, predictions, actuals):
        # Causal analysis of errors
        # When/why does system fail?
        # Returns: insights
        
    def ab_test(self, strategy_a, strategy_b):
        # Automatic A/B testing
        # Bayesian evaluation
        # Keep winner
        
    def curriculum_learn(self):
        # Start with easy examples
        # Gradually increase difficulty
        # Adaptive curriculum
```

#### 4.2 Adaptive Trainer (`adaptive_trainer.py`)
```python
class AdaptiveTrainer:
    """Trains when needed, not on schedule"""
    
    def should_train(self, performance_history):
        # Detects performance degradation
        # Returns: True if training needed
        
    def train_efficiently(self, data):
        # Sample-efficient updates
        # Importance sampling (hard examples)
        # Few-shot learning
        # Transfer learning
```

### Phase 5: Intelligence Modules (src/intelligence/)

#### 5.1 Uncertainty Quantification (`uncertainty.py`)
```python
class UncertaintyEstimator:
    """Knows what it doesn't know"""
    
    def estimate(self, model, x):
        # Bayesian neural network
        # MC Dropout (20 forward passes)
        # Conformal prediction
        # Returns: mean, epistemic_unc, aleatoric_unc
```

#### 5.2 Multi-Task Learning (`multitask.py`)
```python
class MultiTaskPredictor(nn.Module):
    """Predicts multiple objectives"""
    
    def __init__(self):
        self.shared_encoder = AdaptiveEncoder()
        self.price_head = nn.Linear(512, 1)
        self.volatility_head = nn.Linear(512, 1)
        self.regime_head = nn.Linear(512, 5)
        
    def forward(self, x):
        features = self.shared_encoder(x)
        return {
            'price': self.price_head(features),
            'volatility': self.volatility_head(features),
            'regime': self.regime_head(features)
        }
```

#### 5.3 Graph Neural Network (`gnn.py`)
```python
class MarketGNN(torch.nn.Module):
    """Market structure reasoning"""
    
    def __init__(self):
        # Assets = nodes
        # Correlations = edges (dynamic)
        self.conv1 = GCNConv(64, 128)
        self.conv2 = GCNConv(128, 64)
        
    def forward(self, x, edge_index, edge_weight):
        # Message passing over market graph
        # Returns: relationally-aware embeddings
```

#### 5.4 Causal Discovery (`causal.py`)
```python
class CausalDiscovery:
    """Discovers cause-effect relationships"""
    
    def discover_graph(self, data):
        # PC algorithm or neural causal model
        # Returns: DAG of causal relationships
        
    def estimate_effect(self, treatment, outcome):
        # Causal effect estimation
        # DoWhy + EconML
```

### Phase 6: Single Entry Point (run.py)

```python
class EliteSystem:
    """Main system orchestrator"""
    
    def __init__(self, config):
        # Initialize all components
        self.brain = self.build_brain()
        self.data = DataIntelligence()
        self.experience = EpisodicMemory()
        self.improvement = SelfImprovement()
        
    def run_adaptive(self):
        """Main adaptive learning loop"""
        while True:
            # 1. Collect new data
            data = self.data.collect()
            
            # 2. Make prediction
            pred, confidence = self.brain.predict(data)
            
            # 3. RL decision
            action = self.rl_agent.select_action(state)
            
            # 4. Wait for actual outcome
            actual = wait_for_outcome()
            
            # 5. Remember experience
            self.experience.remember(data, pred, actual)
            
            # 6. Online update
            self.brain.online_update(data, actual)
            
            # 7. Self-improve
            if self.improvement.should_train():
                self.brain.meta_train()
                
            # 8. Display
            self.display_status()
            
    def run_nas(self, budget):
        """Neural architecture search"""
        best_arch = self.nas.search(budget)
        self.brain.update_architecture(best_arch)
```

## Requirements (requirements.txt)

```
torch>=2.0.0
torch-geometric>=2.3.0
higher>=0.2.1
stable-baselines3>=2.0.0
optuna>=3.0.0
nni>=2.10
dowhy>=0.9
econml>=0.14
onnxruntime>=1.15.0
h5py>=3.8.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.0
pyyaml>=6.0
rich>=13.0.0
```

## Configuration (config.yaml)

```yaml
system:
  mode: adaptive
  
brain:
  encoder_dim: 512
  attention_heads: 16
  memory_slots: 256
  meta_lr: 0.001
  online_lr: 0.0001
  
rl:
  algorithm: ppo
  gamma: 0.99
  
nas:
  frequency: weekly
  budget: 100
  
experience:
  capacity: 100000
  decay: 0.99
```

## Success Criteria

✅ **Meta-learning works**: Adapts to new regime in <10 examples
✅ **Online learning works**: Updates continuously without forgetting
✅ **Causal discovery works**: Identifies real causal relationships
✅ **RL works**: Makes intelligent decisions (when to predict)
✅ **NAS works**: Improves architecture over time
✅ **Performance**: >70% directional accuracy sustained
✅ **Efficiency**: <20ms inference, <4GB RAM
✅ **Self-improvement**: +1-2% accuracy per month

## What Makes This Elite

1. **Meta-learning**: Learns in minutes (MAML)
2. **Causal**: Understands why, not just correlations
3. **Online**: Never stops evolving
4. **RL**: Optimal decision making
5. **NAS**: Self-designing architecture
6. **Memory**: Episodic recall like humans
7. **Multi-task**: Predicts multiple objectives
8. **GNN**: Understands market structure
9. **Uncertainty**: Knows its confidence
10. **Counterfactual**: Learns from hypotheticals

## Implementation Order

1. ✅ Core encoder (self-attention)
2. ✅ Memory network (read/write)
3. ✅ Online learner (streaming updates)
4. ✅ Meta-learner (MAML)
5. ✅ RL agent (PPO)
6. ✅ Data pipeline (unified)
7. ✅ Experience system (episodic memory)
8. ✅ Causal attention
9. ✅ NAS (run weekly)
10. ✅ Run.py (orchestrate everything)

## Code Quality

- **Type hints everywhere**
- **Docstrings with math notation**
- **Clean, readable, elegant**
- **~4,500 lines total** (not 20,000+)
- **Every line must be intelligent**

## Critical Notes

⚠️ **This is NOT an ensemble** - ONE brain
⚠️ **This is NOT batch training** - online learning
⚠️ **This is NOT manual features** - learned representations
⚠️ **This is NOT correlation** - causal understanding
⚠️ **This is NOT fixed** - self-evolving

This system would be competitive at Renaissance Technologies. Build it that way.

🚀 **Make it brilliant.**
