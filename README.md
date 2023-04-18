# EE-399: Introduction to Machine Learning
#### HW 2 Submission
#### Sabrina Hwang

## Abstract:
This code was created for EE 399 Introduction to Machine Learning, HW 2 submission by Sabrina Hwang. 
This code uses a file that has a total of 39 different faces with about 65 lighting scenes for each 
face (2414 faces in all) in the form of a matrix. The individual images of the columns in the matrix 
X, where each image has been downsampled to 32 x 32 pixels and coverted into greyscale with values 
between 0 and 1. 


The accompanying Python code performs correlation matrix computations, and computes the correlation 
between the set of images provided within the matrix. In addition, the code makes additional 
computations for SVD, eigenvector comparisons, and percentage of variance. 

## Table of Contents:
* [Abstract](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#abstract)
* [Introduction and Overview](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#introduction-and-overview)
* [Theoretical Background](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#theoretical-background)
* [Algorithm Implementation and Development](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#algorithm-implementation-and-development)
  * [Code Description](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#code-description)
    * [Problem 1: Finding Minimum Error and Optimizing Parameters](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-1-finding-minimum-error-and-optimizing-parameters)
    * [Problem 2: Generating 2D Error Landscape](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-2-generating-2d-error-landscape)
    * [Problem 3: Fitting and Applying Models to Datasets I](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-3-fitting-and-applying-models-to-datasets-i)
    * [Problem 4: Fitting and Applying Models to Datasets II](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-4-fitting-and-applying-models-to-datasets-ii)
* [Computational Results](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#computational-results)
  * [Usage](https://github.com/hwangsab/EE-399/blob/main/README.md#usage)
  * [Problem 1: Finding Minimum Error and Optimizing Parameters](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-1-finding-minimum-error-and-optimizing-parameters-1)
  * [Problem 2: Generating 2D Error Landscape](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-2-generating-2d-error-landscape-1)
  * [Problem 3: Fitting and Applying Models to Datasets I](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-3-fitting-and-applying-models-to-datasets-i-1)
  * [Problem 4: Fitting and Applying Models to Datasets II](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#problem-4-fitting-and-applying-models-to-datasets-ii-1)
* [Summary and Conclusion](https://github.com/hwangsab/EE-399-HW2/blob/main/README.md#summary-and-conclusions)

## Introduction and Overview:
In this homework assignment, we will explore a dataset of 39 different faces, each with about 65 
different lighting scenes, for a total of 2,414 images. The images are downsampled to 32x32 pixels 
and converted into grayscale with values ranging from 0 to 1. The dataset is stored in a matrix X of 
size 1024x2414, where each column corresponds to an image.

The first task (a) is to compute a 100x100 correlation matrix C, where each element represents the 
correlation between two images. The correlation is computed as the dot product between the two 
images' vectors. We will plot the correlation matrix using the pcolor function.

From the correlation matrix, we will identify the two images that are most highly correlated and the 
two images that are least correlated in part (b). We will plot these faces to visually compare the 
similarities and differences between them.

Next, we will repeat the correlation matrix computation in (c), but this time we will compute a 10x10 
matrix and plot it. This will allow us to compare the correlation between images in a smaller subset 
of the dataset.

In parts (d) and (e), we will use different methods to find the first six eigenvectors and principal 
component directions, respectively, of the matrix X. We will then compare the first eigenvector found 
using both methods and compute the norm of the difference in their absolute values.

Finally, we will compute the percentage of variance captured by each of the first six SVD modes and 
plot the first six SVD modes. This will give us an idea of how much information is retained by using 
these modes to represent the images instead of the original matrix X.

## Theoretical Background:
To successfully complete this assignment, a solid foundation in linear algebra and its applications 
is necessary. Specifically, knowledge of matrix operations, eigenvalues and eigenvectors, and 
singular value decomposition (SVD) is essential.

Matrix operations such as addition, subtraction, and multiplication are fundamental to this 
assignment. We will use dot products and element-wise multiplication of matrices to compute the 
correlation matrix and to find the eigenvectors and singular values of the matrix X. Additionally, 
knowledge of matrix transposition and reshaping will be useful in preparing the data for analysis and 
visualization.

Eigenvalues and eigenvectors play a critical role in analyzing the correlation matrix and finding the 
first six eigenvectors with the largest magnitude eigenvalue. Eigenvectors represent directions along 
which a transformation acts only by stretching or shrinking, and eigenvalues represent the magnitude 
of the stretch or shrink in that direction. In this assignment, we will use the eigenvectors of the 
correlation matrix to find the principal components of the data, which can be used to reduce the 
dimensionality of the dataset while retaining most of the information.

Singular value decomposition (SVD) is another key concept for this assignment. SVD decomposes a 
matrix into three components: U, Σ, and V*, where U and V* are orthogonal matrices and Σ is a 
diagonal matrix with the singular values of the original matrix on its diagonal. SVD is useful for 
finding the principal components of a dataset and for compressing the data by retaining only a subset 
of the singular values.

## Algorithm Implementation and Development:
This homework assignment works around the following dataset:
```
X = np.arange(0, 31)
Y = np.array([30, 35, 33, 32, 34, 37, 39, 38, 36, 36, 37, 39, 42, 45, 45, 41, 40, 39, 42, 44, 47, 49, 50, 49, 46, 48, 50, 53, 55, 54, 53])
```

Completion of this project and subsequent development and implementation of the algorithm was 
accomplished through Python as our primary programming language. 

### Code Description
The code is written in Python and uses the following libraries:  
* `numpy` for numerical computing  
* `matplotlib` for data visualization  
* `math` for mathematical functions  
* `warnings` for error message override  
* `scipy` for curve fitting  

#### Problem 1: Finding Minimum Error and Optimizing Parameters
The code reads a dataset of 31 points and defines a function to fit the data using least-squares curve 
fitting. The function `func(X, A, B, C, D)` is a combination of a cosine function and a linear function 
with four parameters $A$, $B$, $C$, $D$ that are to be optimized. The `curve_fit` function from scipy 
library is used to find the optimized values of the parameters. Then, the minimum error between the 
function and the dataset is calculated, and the results are printed along with a plot of the function 
fit to the data. 

```
def func(x, A, B, C, D):
    return A*np.cos(B*x) + C*x + D

popt, pcov = curve_fit(func, X, Y)

A, B, C, D = popt

error = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))
```

#### Problem 2: Generating 2D Error Landscape
The code generates a 2D error landscape by sweeping through different values of the function 
parameters and fixing two parameters at a time. The error is calculated for each combination of 
parameter values, and the results are plotted using pcolor from matplotlib library. 

The code first fixes A and B parameters and sweeps through C and D parameters, then fixes A and C 
parameters and sweeps through B and D parameters, and finally fixes A and D parameters and sweeps 
through B and C parameters. The min function is used to find the minimum error and the corresponding 
parameter values. 

For the example of a fixed parameters A and B, sweeping C and D program:
```
C_range = np.linspace(-5, 5, 100)
D_range = np.linspace(30, 60, 100)
C_grid, D_grid = np.meshgrid(C_range, D_range)
error_grid = np.zeros_like(C_grid)
for i in range(len(C_range)):
    for j in range(len(D_range)):
        C = C_range[i]
        D = D_range[j]
        error_grid[j,i] = np.sqrt(np.mean((func(X, A, B, C, D) - Y)**2))
```

#### Problem 3: Fitting and Applying Models to Datasets I
The code uses the first 20 data points as training data to fit a line, a parabola, and a 19th degree
polynomial to the specified points. Then, after computing the least-square error for each of these
models over the training points, the program then computes the least-square error for each of these
models on the remaining 10 data points excluded from the training data, which we refer to in the code
as the test data. 

```
# fit line, parabola, and 19th degree polynomial
line_coeffs = np.polyfit(X_train, Y_train, 1)
parabola_coeffs = np.polyfit(X_train, Y_train, 2)
poly_coeffs = np.polyfit(X_train, Y_train, 19)

# compute predictions on train and test data
Y_line_train = np.polyval(line_coeffs, X_train)
Y_parabola_train = np.polyval(parabola_coeffs, X_train)
Y_poly_train = np.polyval(poly_coeffs, X_train)

Y_line_test = np.polyval(line_coeffs, X_test)
Y_parabola_test = np.polyval(parabola_coeffs, X_test)
Y_poly_test = np.polyval(poly_coeffs, X_test)

# compute least square error on train and test data
line_train_error = np.sum((Y_line_train - Y_train)**2)
parabola_train_error = np.sum((Y_parabola_train - Y_train)**2)
poly_train_error = np.sum((Y_poly_train - Y_train)**2)

line_test_error = np.sum((Y_line_test - Y_test)**2)
parabola_test_error = np.sum((Y_parabola_test - Y_test)**2)
poly_test_error = np.sum((Y_poly_test - Y_test)**2)
```

#### Problem 4: Fitting and Applying Models to Datasets II
The code uses the first 10 and last 10 data poitns as training data to fit a line, a parabola, and a 
19th degree polynomial to the specified points. Then, after computing the least-square error for each 
of these models over the training points, the program then computes the least-square error for each of 
these models on the remaining 10 data points from the middle of the data set, which were excluded from 
the training data, which we refer to in the code as the test data. 

## Computational Results:

### Usage
To run the code, simply run the Python file hw1.py in any Python environment. The output will be 
printed to the console and displayed in a pop-up window. The matplotlib library is required to display 
the 2D error landscape plot. 

#### Problem 1: Finding Minimum Error and Optimizing Parameters
The resultant cosine model fits over the data with optimized parameters with values as follows:
```
A = 2.1717269828948855
B = 0.909325796914226
C = 0.7324885143513572
D = 31.452772437053802
```
In addition, the model has an minimum error value of `1.5927258503103892`

![Q1](https://user-images.githubusercontent.com/125385468/231071680-c452328b-7c99-4d80-91c0-8577614a15a9.png)

#### Problem 2: Generating 2D Error Landscape
The resultant minimum error calculated is as follows:
```
Fixed parameters A and B Minimum error: 1.61 at C = -4.60, D = 47.27
Fixed parameters A and C Minimum error: 72.16 at B = 0.01, D = 33.64
Fixed parameters A and D Minimum error: 14.69 at B = 0.44, C = 3.89
Fixed parameters B and C Minimum error: 73.17 at A = 0.10, D = 60.00
Fixed parameters B and D Minimum error: 14.75 at A = 0.93, C = -5.00
Fixed parameters C and D Minimum error: 99.23 at A = 0.35, B = 1.00
```

2D Loss Landscapes are plotted for all 6 combinations as followed:
![Q2](https://user-images.githubusercontent.com/125385468/231071717-11467706-e529-4f1d-bba3-6471f5920e45.png)

#### Problem 3: Fitting and Applying Models to Datasets I
The following graph displays the first 20 data points used as training data for a line, a parabola, and a 19th 
degree polynomial model fit. 

![Q3](https://user-images.githubusercontent.com/125385468/231071742-94d348a2-2248-4adf-828d-f3e9af9a42b6.png)

```
Line Train Error: 100.59849624060148
Parabola Train Error: 90.35835042150828
19th Degree Polynomial Train Error: 0.016076685015099894

Line Test Data Error: 124.45472101305883
Parabola Test Data Error: 835.2050011334942
19th Degree Polynomial Test Data Error: 9.013397831469909e+21
```

#### Problem 4: Fitting and Applying Models to Datasets II
The following graph displays the first 10 and last 10 data points used as training data for a line, a parabola, 
and a 19th degree polynomial model fit. 

![Q4](https://user-images.githubusercontent.com/125385468/231071779-fa8dc3e8-6866-49ca-932b-f11e48f9b65c.png)

```
Line Train Error: 68.75016908693057
Parabola Train Error: 68.73967050163469
19th Degree Polynomial Train Error: 1.0566927249586058

Line Test Data Error: 86.95136045541048
Parabola Test Data Error: 86.16001918838907
19th Degree Polynomial Test Data Error: 67127.9806069814
```
Comparing questions II.iii and II.iv, the model generated that takes in the first 10 and last 10 data points 
as training data and the middle 10 data points as test data (from II.iv) has a lower minimized error. The 
magnitude at which these errors are different is that as the polynomial degree increases, the minimized 
error will decrease more. This is most likely because the model accounts for the shape of the beginning and 
the end, and because the dataset provided is relatively continuous, the model fits the data better if it is 
trained with points from the beginning and the end.

## Summary and Conclusions:
In this assignment, we learned how to perform data analysis on a set of images using linear algebra 
concepts and Python programming. Specifically, we computed the correlation matrix between images, 
found the most highly correlated and uncorrelated images, computed the eigenvectors and singular 
values of the matrix, and analyzed the percentage of variance captured by the first six SVD modes.

We applied our knowledge of matrix operations, eigenvalues and eigenvectors, and singular value 
decomposition to perform these analyses. We used dot products, element-wise multiplication, and 
reshaping of matrices to prepare the data for analysis and visualization. We computed the eigenvalues 
and eigenvectors of the correlation matrix to find the principal components of the data. We also 
performed SVD to compress the data by retaining only a subset of the singular values.

In conclusion, this assignment provided a valuable opportunity to apply theoretical concepts to real-
world data and gain practical experience in data analysis. The knowledge and skills gained from this 
assignment can be applied to a wide range of fields and applications, such as image processing, 
machine learning, and data compression. By continuing to explore and expand upon these concepts and 
techniques, we can continue to develop our understanding and expertise in linear algebra and data 
analysis. 
