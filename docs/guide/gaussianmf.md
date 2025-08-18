### Gaussian Membership Function

The **Gaussian membership function** (GaussianMF) is a fundamental concept in fuzzy logic, widely used to define fuzzy sets. Unlike a classical set where an element either fully belongs or does not belong, a fuzzy set allows for partial membership, and the GaussianMF provides a smooth, continuous way to represent this degree of belonging. It possesses a **smooth, bell-like shape**, which contributes to its intuitive nature and popularity across various applications.

The function is characterized by two main parameters:

  * **mean ($\mu$)**: This parameter determines the **center** of the curve. It represents the point in the domain where the degree of membership is maximum, specifically 1.
  * **sigma ($\sigma$)**: This parameter controls the **width** or spread of the curve. It must be a positive value. A larger $\sigma$ results in a wider, flatter curve, indicating a broader range of values with high membership. Conversely, a smaller $\sigma$ produces a sharper, more peaked curve, suggesting a narrower range of values with a high degree of belonging.

The mathematical formula for the Gaussian membership function is given by:

$$\mu(x) = e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

where:

  * $\mu(x)$ is the degree of membership of element $x$ in the fuzzy set.
  * $x$ is the input value.
  * $\mu$ is the mean of the curve.
  * $\sigma$ is the standard deviation (width) of the curve.

-----

### Partial Derivatives

The partial derivatives of the Gaussian membership function are crucial for optimization algorithms, especially in adaptive or machine learning-based fuzzy systems. They show how the membership value changes in response to small adjustments to the parameters, which is essential for training models to better fit data.

#### Derivative with respect to $\mu$

The partial derivative of the function with respect to the mean ($\mu$) is:

$$\frac{\partial f}{\partial \mu} = \frac{x-\mu}{\sigma^2} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

This derivative indicates how the membership value is affected when the center of the bell curve is shifted. It is used to adjust the position of the function to better align with the data.

#### Derivative with respect to $\sigma$

The partial derivative with respect to the standard deviation ($\sigma$) is:

$$\frac{\partial f}{\partial \sigma} = \frac{(x-\mu)^2}{\sigma^3} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

This derivative shows how the membership value changes as the width of the curve is adjusted. It is used to refine the spread of the function, making it sharper or wider as needed to represent the uncertainty in the data more accurately.

-----

### Python Example

The following code snippet demonstrates how to generate a Gaussian membership function using the **`numpy`** and **`matplotlib`** libraries in Python.

```python
import numpy as np
from anfis_toolbox import GaussianMF
import matplotlib.pyplot as plt


# Generate a range of x values
x = np.linspace(0, 10, 100)

# Create a Gaussian membership function with a mean of 5 and a sigma of 2
gaussian = GaussianMF(5, 1)

# Calculate the membership values (y) for each x value
y = gaussian(x)

plt.plot(x, y)
plt.show()
```
 <img src="/anfis-toolbox/assets/gaussian_mf.svg" alt="gaussian" />

-----

### Visualization

Below is a visual representation of a Gaussian membership function, showing how its shape is influenced by the **mean ($\mu$)** and **sigma ($\sigma$)** parameters.

<iframe src="/anfis-toolbox/assets/gaussian_mf_subplots.html" width="100%" height="550" frameborder="0" loading="lazy"></iframe>

The image displays a classic bell-shaped curve, illustrating how the membership value (on the y-axis) smoothly changes for different input values (on the x-axis). The peak of the curve is located at the mean, and the spread of the curve is controlled by sigma.
