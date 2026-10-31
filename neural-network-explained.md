# Live Neural Network: How It Works

**Designed and created by Gregory Gasowski**

This document explains every part of the live neural network animation: what you see on screen, the math behind it, how the model trains, and why each design choice was made.

The demo is three files:

| File | Purpose |
|---|---|
| `neural-network.html` | The page: background photo, header, step panel, loss chart, hover box, sound button |
| `neural-network.js` | All logic: the model, training, animation, interaction and sound |
| `background.jpg` | The background photo |

Open `neural-network.html` in any modern browser. Nothing needs to be installed and nothing is loaded from the internet.

---

## 1. Overview

- **Network:** 2 inputs (x₁, x₂), 10 hidden layers of 5 neurons each using ReLU, and 1 output (ŷ). One output gives a single prediction to measure the loss against.
- **Hover:** point at any neuron (or tap it on a phone) and a box shows its actual calculation with the current numbers. For a first-layer neuron that is exactly the classic example: z = (x₁·w₁) + (x₂·w₂) + b, filled in with the real values, then a = ReLU(z) = max(0, z). Signals flow into that neuron continuously and each incoming connection shows its weight. After backpropagation, the box also shows that neuron's error (δ) and its gradient.
- **Training steps:** a panel along the bottom lights up each step in turn, with live numbers:
  1. **Forward propagation:** signals travel left to right, and neurons that ReLU switches off stay dark.
  2. **Loss calculation:** L = ½(ŷ − y)², with the output neuron flashing.
  3. **Backpropagation:** the error travels right to left as amber pulses.
  4. **Optimization:** weights update using w ← w − η·∂L/∂w, and the connections that changed the most flash white.
- **Training loss chart:** shows the loss trend as the network learns.

**Two things to know:**

- **The training is real.** Each cycle animates one example in detail, and behind the scenes it runs 10 small batches of training updates. In testing, the loss usually drops clearly within 3 to 4 minutes.
- **Some runs learn slowly.** A network this deep and narrow with ReLU occasionally starts badly (about 1 run in 8 in testing). The code restarts automatically if it stops learning completely, but a slow run just keeps going, so reload the page if the chart stays flat.

---

## 2. Configuration

All settings live in the `CONFIG` object at the top of `neural-network.js`.

| Setting | Value | What it does and why |
|---|---|---|
| `inputs` | 2 | Two input features, x₁ and x₂. Two is the smallest number that still makes the classic formula z = (x₁·w₁) + (x₂·w₂) + b visible in the first hidden layer. |
| `hiddenLayers` | 10 | Ten hidden layers make the signal visibly travel a long way, which shows how depth works. |
| `neuronsPerHidden` | 5 | Five neurons per layer keeps every connection readable on screen (25 connections between two hidden layers). |
| `outputs` | 1 | One prediction, ŷ, so the loss is a single clear number. |
| `learningRate` (η) | 0.05 | Step size for each weight update. Chosen in testing: lower values learned too slowly to see; higher values became unstable. |
| `batchSize` | 32 | Examples averaged in each training update. Averaging smooths out noise from individual examples. |
| `updatesPerCycle` | 10 | Training updates run during each animated cycle, so learning is visible within minutes rather than hours. |
| `stepMs` | 260 | Milliseconds for a signal to travel one layer. |
| `lossMs` | 1100 | Duration of the loss step. |
| `updateMs` / `pauseMs` | 900 / 350 | Duration of the optimization flash, then a short pause before the next example. |
| `target(x₁, x₂)` | 0.8·x₁ − 0.5·x₂ + 0.3·x₁·x₂ + 0.2 | The hidden "true" function the network learns. It includes a product term (x₁·x₂), so it is not a straight line and the hidden layers have real work to do. |
| `colors` | teal, blue, amber, white | Teal = positive weight or activation, blue = negative weight, amber = error/gradient, white = weight update. |

One full cycle takes about 8 seconds: forward 2.9 s (11 layer hops × 260 ms), loss 1.1 s, backpropagation 2.9 s, optimization 1.25 s.

---

## 3. The Model

### Architecture

```
Input (2) → H1 (5) → H2 (5) → … → H10 (5) → Output (1)
```

Every neuron is connected to every neuron in the next layer (fully connected). That gives:

- Weights: 2×5 + 9×(5×5) + 5×1 = **240 weights**
- Biases: 10×5 + 1 = **51 biases**
- **291 trainable parameters** in total

### Data

Each training example is generated on the fly:

- x₁ and x₂ are random numbers between −1 and 1
- the correct answer is y = target(x₁, x₂)

Because examples are fresh every time, the network can't simply memorize them. It has to learn the function itself.

### Weight initialization (He initialization)

Each weight starts as a random number from a normal distribution with standard deviation √(2 / n), where n is the number of inputs to that layer.

**Why:** ReLU sets every negative value to zero, so roughly half the signal is lost at each layer. He initialization scales the starting weights to compensate, so the signal keeps a healthy size through all 10 layers instead of shrinking to nothing or growing out of control.

### Bias initialization

Every bias starts at 0.05.

**Why:** a small positive bias nudges each neuron slightly toward being active at the start. That reduces the chance of "dead" neurons (see section 9).

### Activation function: ReLU

Hidden neurons use ReLU (Rectified Linear Unit):

```
a = ReLU(z) = max(0, z)
```

If z is positive, it passes through unchanged. If z is zero or negative, the neuron outputs 0 and is "switched off". On screen, switched-off neurons stay dark during the forward pass.

**Why ReLU:** it is the standard activation for deep networks. It is fast to compute and its gradient is simply 1 (active) or 0 (inactive), which helps the error signal travel back through many layers.

### Output neuron: linear

The output neuron does not use ReLU: ŷ = z.

**Why:** the target can be negative. A ReLU output could never predict a negative number.

---

## 4. Step 1: Forward Propagation

For every neuron j in a layer, using the activations a from the previous layer:

```
z_j = (a₁·w₁ⱼ) + (a₂·w₂ⱼ) + … + (aₙ·wₙⱼ) + b_j
a_j = ReLU(z_j)
```

For the first hidden layer the previous activations are the inputs themselves, so this becomes the classic form:

```
z = (x₁·w₁) + (x₂·w₂) + b
```

**Worked example** (numbers like those shown live on screen):

```
x₁ = −0.40, x₂ = 0.87, w₁ = 0.22, w₂ = −0.04, b = 0.14
z = (−0.40 × 0.22) + (0.87 × −0.04) + 0.14
  = −0.088 − 0.035 + 0.14
  = 0.017 ≈ 0.02
a = ReLU(0.02) = 0.02   (positive, so the neuron is active)
```

**On screen:**

- A pulse travels along each connection, one layer at a time. Its size and brightness show the signal it carries, aᵢ·w. Very weak signals (below 0.02) are not drawn.
- When the wave reaches a layer, each neuron glows according to its activation. Neurons switched off by ReLU stay dark.
- Connection color shows the weight's sign (teal positive, blue negative). Thickness shows its strength.

---

## 5. Step 2: Loss Calculation

The loss measures how wrong the prediction is:

```
L = ½ (ŷ − y)²
```

This is squared error. The ½ is a convention that makes the derivative cleaner.

**Example:** ŷ = 0.14 and y = −0.67 gives L = ½ × (0.81)² = 0.328.

**Why squared error:** it is the standard loss for predicting a number. It is always positive, it is zero only when the prediction is exact, and it penalizes big mistakes much more than small ones.

**On screen:** the output neuron pulses amber, and the panel shows ŷ, y and L.

---

## 6. Step 3: Backpropagation

Backpropagation works out how much each weight contributed to the error. It applies the chain rule of calculus, starting at the output and moving backwards one layer at a time.

**1. Error at the output.** For L = ½(ŷ − y)² with a linear output:

```
δ_out = ∂L/∂z_out = ŷ − y
```

**2. Error in each earlier neuron i.** It collects the errors of the neurons it feeds, weighted by the connections, then passes through ReLU's derivative:

```
δᵢ = ( Σⱼ wᵢⱼ · δⱼ ) × ReLU′(zᵢ)

ReLU′(z) = 1 if z > 0, otherwise 0
```

So a neuron that was switched off in the forward pass gets zero error. The hover box shows this as "ReLU blocks gradient".

**3. Gradient for every weight and bias:**

```
∂L/∂wᵢⱼ = aᵢ · δⱼ        (input to the connection × error at its end)
∂L/∂bⱼ  = δⱼ
```

**On screen:** amber pulses travel right to left. Their size shows the error each connection carries back, |δⱼ · w|. Neurons light amber as the error reaches them.

---

## 7. Step 4: Optimization (Mini-batch SGD)

Each weight moves a small step in the direction that reduces the loss:

```
w ← w − η · ∂L/∂w
b ← b − η · ∂L/∂b
```

with learning rate η = 0.05.

### Mini-batches

Instead of updating from one example at a time, each update averages the gradients of 32 examples (`batchSize`). The example animated on screen is included in the first batch of each cycle.

**Why:** one example gives a noisy direction. The average of 32 gives a more reliable one.

### Several updates per cycle

Each animated cycle runs 10 updates (`updatesPerCycle`), so 320 examples per cycle.

**Why:** a cycle takes about 8 seconds so the eye can follow it. With only one update per cycle, the network would need hours to visibly learn. Ten updates per cycle make progress visible within minutes, while the animation still shows one example step by step.

### Gradient clipping

Before each update, every averaged gradient is limited to the range −1 to +1:

```
g = max(−1, min(1, g))
```

**Why:** in a deep network, an unlucky example can produce a very large gradient. One huge update can throw the weights far off and break training (often called "exploding gradients"). Clipping caps the size of any single step and keeps training stable. It is applied to every weight and bias gradient.

**On screen:** the connections whose weights changed the most during the cycle flash white. Brightness is relative to the largest change in that cycle.

---

## 8. Measuring Progress: The Loss Chart

After each cycle, the code measures the **average loss on 64 brand-new random examples** the network has never seen.

**Why fresh examples:** this measures how well the network has learned the function in general, not how well it did on examples it just trained on.

The chart:

- keeps the last 160 measurements
- draws a smoothed trend line, an exponential moving average: smoothed = 0.6 × previous smoothed + 0.4 × new value
- rescales automatically between the lowest and highest values shown

The panel also shows the total number of updates and the latest average loss.

---

## 9. Automatic Restart

Deep, narrow ReLU networks have a known failure mode called **dying ReLU**. If every neuron in one layer becomes permanently inactive, no signal passes forward and no gradient passes back, so training stops completely.

The code restarts with fresh weights if:

- the loss becomes invalid (not a finite number), or
- after 40 cycles, the loss has not changed at all over the last 10 measurements while still above 0.05

A run that is learning slowly is left alone. Only a run that is completely stuck restarts.

---

## 10. Hover: The Algorithm for Each Neuron

Point at a neuron (within about 34 pixels) or tap it to open a box with its real, current calculation.

| Neuron | The box shows |
|---|---|
| Input (x₁, x₂) | The input value |
| Hidden | The formula z = (a₁·w₁) + … + b, the same formula with live numbers, the result z, then a = ReLU(z) = max(0, z) and whether the neuron is inactive |
| Output (ŷ) | The same z calculation, ŷ = z (linear output), the target y, and the loss L |

During and after backpropagation the box adds:

- δ = ∂L/∂z for that neuron, with a note when ReLU blocks the gradient
- an example gradient, ∂L/∂w₁ = δ · a₁ (or δ · x₁ in the first layer)

While hovering, the incoming connections are highlighted, each shows its weight label (for example w₁ = 0.22), and signals flow into the neuron continuously.

---

## 11. Drag Interaction

Any neuron can be grabbed (within about 30 pixels) and dragged. When released, it springs back to its exact home position with a damped wobble.

**Dragging is visual only.** It never changes weights, biases, activations or the neuron's home position. The training continues unaffected.

The spring-back uses simple physics on the offset from home:

```
velocity += (−0.12 × offset − 0.13 × velocity)
offset   += velocity
```

0.12 is the spring stiffness and 0.13 is the damping. Together they give one or two small overshoots before the neuron settles.

Cursor feedback: an open hand over a neuron, a closed hand while dragging. Touch works the same way on phones and tablets.

---

## 12. Sound Design

All sounds are generated live in the browser with the Web Audio API. There are no audio files. The style is inspired by the analog sequencer music of Jean-Michel Jarre and Tangerine Dream.

| Event | Sound |
|---|---|
| Hover over a neuron | A soft, short sine blip. Pitch rises from left to right across the layers. Plays only after the first click or tap, and at most every 90 ms. |
| Grab a neuron | A deep analog "zwoop": two slightly detuned oscillators (square + sawtooth) through a resonant low-pass filter that sweeps open, with the pitch gliding upward. Pitch depends on the layer. |
| Release a neuron | A slow step-sequencer run in a low A minor pentatonic scale, 0.2 s per step. The filter opens a little more on each note. Longer pulls play more steps (4 to 8). |
| Echo | All synth sounds feed a shared echo: 0.46 s delay, 58% feedback, darkened by a low-pass filter at 1500 Hz, for a long, spacey tail. |

The **Sound on / Sound off** button in the top-right corner mutes everything. Browsers require a click before any sound can play, so sound starts after the first interaction.

---

## 13. Layout, Accessibility and Performance

- **Responsive layout:** on wide screens the network flows left to right. When the screen is taller than it is wide (phones in portrait), it flows top to bottom. The network is positioned below the header and above the step panel automatically.
- **Sharp on high-resolution screens:** the canvas renders at up to 2× pixel density.
- **Reduced motion:** visitors whose system is set to reduce motion see a still frame instead of the animation.
- **Battery and CPU:** the animation pauses when the browser tab is hidden.
- **Background glows:** two soft color glows drift slowly behind the network, purely decorative.

---

## 14. Customizing

Change any value in `CONFIG` at the top of `neural-network.js`, for example:

- more or fewer layers or neurons: `hiddenLayers`, `neuronsPerHidden`
- faster or slower animation: `stepMs`, `lossMs`, `updateMs`
- faster or more cautious learning: `learningRate`, `updatesPerCycle`
- a different function to learn: `target`

For testing in the browser console, `window.nnDemo` exposes the model functions (`forward`, `backward`, `optimize`, `evalLoss`, `sample`), the configuration and the training state.

---

*Live Neural Network. Designed and created by Gregory Gasowski. [gregorygasowski.github.io](https://gregorygasowski.github.io)*
