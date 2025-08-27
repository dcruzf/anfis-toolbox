### Generalized Bell Membership Function

The **Generalized Bell membership function** (GbellMF), also known as the **Bell-shaped curve**, is a versatile function used in fuzzy logic to define fuzzy sets. Like other membership functions, it assigns a degree of membership to an element, but it offers greater flexibility than the GaussianMF by using an additional parameter to control its shape. Its form is a smooth, symmetrical bell curve.

The function is defined by three parameters:

  * **center** ($c$): This parameter determines the **center** of the curve, representing the point in the domain with a maximum membership value of 1.
  * **width ($a$)**: This parameter controls the **width** or spread of the curve. A larger value of $a$ results in a wider curve, while a smaller value produces a narrower curve.
  * **slope ($b$)**: This parameter, which must be a positive value, determines the **slope** of the curve's sides. It directly impacts the steepness of the curve's transition from 0 to 1. A larger $b$ value creates a steeper curve, making the fuzzy set sharper and less "fuzzy."

The mathematical formula for the Generalized Bell membership function is given by:

$$\mu(x) = \frac{1}{1 + \left|\frac{x-c}{a}\right|^{2b}}$$

where:

  * $\mu(x)$ is the degree of membership of element $x$ in the fuzzy set.
  * $x$ is the input value.
  * $c$ is the center of the curve.
  * $a$ is the width of the curve.
  * $b$ is the slope of the curve.

-----

### Partial Derivatives

The partial derivatives of the GbellMF are essential for training and optimizing fuzzy systems. They show how the membership value changes with respect to small changes in each of the three parameters, which is vital for algorithms like backpropagation used in fuzzy-neural networks.

#### Derivative with respect to the center ($c$)

The partial derivative of the function with respect to its center ($c$) is:

$$\frac{\partial f}{\partial c} = \frac{2b(x-c)}{a^2} \left(1 + \left|\frac{x-c}{a}\right|^{2b}\right)^{-2}$$

This derivative indicates how the membership value changes when the curve is shifted along the x-axis.

#### Derivative with respect to the width ($a$)

The partial derivative with respect to the width ($a$) is:

$$\frac{\partial f}{\partial a} = \frac{2b(x-c)^2}{a^3} \left(1 + \left|\frac{x-c}{a}\right|^{2b}\right)^{-2}$$

This derivative helps in adjusting the spread of the fuzzy set to encompass a broader or narrower range of values.

#### Derivative with respect to the slope ($b$)

The partial derivative with respect to the slope ($b$) is:

$$\frac{\partial f}{\partial b} = -\frac{2}{b} \frac{(x-c)^2}{a^2} \left(\frac{1}{1 + \left|\frac{x-c}{a}\right|^{2b}}\right)^2 \ln\left|\frac{x-c}{a}\right|$$

This derivative is used to modify the steepness of the curve, allowing for fine-tuning of the transition from non-membership to full membership.

-----

### Python Example

The following code demonstrates how to generate a Generalized Bell membership function using the **`numpy`** and **`matplotlib`** libraries in Python.

```python
import numpy as np
import matplotlib.pyplot as plt
from anfis_toolbox import BellMF

bellmf = BellMF(a=2, b=4, c=5)
x = np.linspace(0, 10, 100)
y = bellmf(x)

plt.plot(x, y)
plt.show()
```
<img src="/anfis-toolbox/assets/bell_mf.svg" alt="gaussian" />

-----

### Visualization

Below is a visual representation of a Generalized Bell membership function, showing how its shape is influenced by the **width ($a$)** and **slope ($b$)** parameters, while the **center ($c$)** remains fixed.


<iframe src="/anfis-toolbox/assets/bell_mf_subplots.html" width="100%" height="550" frameborder="0" loading="lazy"></iframe>



The image displays a series of symmetrical bell-shaped curves. You can observe how increasing the width ($a$) makes the curve broader, while increasing the slope ($b$) makes the sides of the curve steeper.
