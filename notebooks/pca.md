# What do Principal Components Actually do Mathematically?
I have recently taken an interest in PCA after watching Professor Gilbert Strang’s [PCA lecture](https://www.youtube.com/watch?v=Y4f7K9XF04k). I must have watched at least 15 other videos and read 7 different blog posts on PCA since. They are all very excellent resources, but I found myself somewhat unsatisfied. What they do a lot is teaching us the following:
- What the PCA promise is;
- Why that promise is very useful in Data Science; and
- How to extract these principal components. (Although I don't agree with how some of them do it by applying SVD on the covariance matrix, that can be saved for another post.)

Some of them go the extra mile to show us graphically or logically, how the promise is being fulfilled by the principal components:
- Graphically, a transformed vector can be shown to be still clustered with its original group in a plot; and
- Logically, a proof can be expressed in mathematical symbols.

## Objective
My mind was convinced, but my heart was still not. To me, the graphic approach does not provide a visual effect that is striking enough. The logic approach, on the other hand, does not even spend enough effort to explain what precisely it's trying to prove. Therefore, the objective of this post is to improve both of these 2 areas - to provide a more striking visual and to establish a more precise goal for our mathematical proof.

## Prerequisites
This post is for you if:
- You have at least a slight idea of what the PCA promise is;
- You have a decent understanding of what the covariance matrix is about;
- You have a solid foundation in linear algebra, e.g., comfortable with the concept of matrices being transformations that rotate and/or stretch spaces; and
- Your heart is longing to discover the principal components, instead of being told what they are!

## How to Choose P?
After hearing my dissatisfaction, my friend [Calvin](https://calvinfeng.github.io/) recommended this paper by Jonathon Shlens - [A Tutorial on Principal Component Analysis](https://arxiv.org/pdf/1404.1100.pdf) to me. It is by far the best resource I have come across on PCA. However, it's also a bit lengthier than your typical blog post, so the remainder of this post will focus on section 5 of the paper. In there, Jonathon immediately establishes the following goal:
> The [original] dataset is $X$, an $m × n$ matrix.<br />
> Find some orthonormal matrix $P$ in $Y = PX$ such that $C_Y \equiv \frac{1}{n}YY^T$ is a diagonal matrix.[1]<br />
> The rows of $P$ shall be principal components of $X$.

As you might have noticed, $C_Y$ here is the covariance matrix of our rotated dataset $Y$. Why do we want $C_Y$ to be diagonal? Before we answer this question, let’s generate a dataset $X$ consisting of 4 features with some random values.


<details>
  <summary>Skippable imports and helpers</summary>

```python
from IPython.display import Latex, display
from string import ascii_lowercase
import numpy as np
import pandas as pd

# constants
FEAT_NUM, SAMPLE_NUM = 4, 4

# helpers
def covariance_matrix(dataset):
    return dataset @ dataset.transpose() / SAMPLE_NUM

def tabulate(dataset, rotated=False):
    '''
    Label row(s) and column(s) of a matrix by wrapping it in a dataframe.
    '''
    if rotated:
        prefix = 'new_'
        feats = ascii_lowercase[FEAT_NUM:2 * FEAT_NUM]
    else:
        prefix = ''
        feats = ascii_lowercase[0:FEAT_NUM]
    return pd.DataFrame.from_records(dataset,
                                     columns=['sample{}'.format(num) for num in range(SAMPLE_NUM)],
                                     index=['{}feat_{}'.format(prefix, feat) for feat in feats])

def display_df(dataset, latex=False):
    rounded = dataset.round(15)
    if latex:
        display(Latex(rounded.to_latex()))
    else:
        display(rounded)
```
</details>

```python
x = tabulate(np.random.rand(FEAT_NUM, SAMPLE_NUM))
display_df(x)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample0</th>
      <th>sample1</th>
      <th>sample2</th>
      <th>sample3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat_a</th>
      <td>0.472612</td>
      <td>0.453242</td>
      <td>0.811147</td>
      <td>0.237625</td>
    </tr>
    <tr>
      <th>feat_b</th>
      <td>0.728994</td>
      <td>0.916212</td>
      <td>0.202783</td>
      <td>0.116406</td>
    </tr>
    <tr>
      <th>feat_c</th>
      <td>0.803590</td>
      <td>0.967202</td>
      <td>0.659594</td>
      <td>0.726142</td>
    </tr>
    <tr>
      <th>feat_d</th>
      <td>0.771849</td>
      <td>0.753178</td>
      <td>0.153215</td>
      <td>0.459026</td>
    </tr>
  </tbody>
</table>
</div>


$X$ above just looks like a normal dataset. Nothing special. What about its covariance matrix?


```python
c_x = covariance_matrix(x)
display_df(c_x)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feat_a</th>
      <th>feat_b</th>
      <th>feat_c</th>
      <th>feat_d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feat_a</th>
      <td>0.285804</td>
      <td>0.237986</td>
      <td>0.381435</td>
      <td>0.234878</td>
    </tr>
    <tr>
      <th>feat_b</th>
      <td>0.237986</td>
      <td>0.356387</td>
      <td>0.422564</td>
      <td>0.334312</td>
    </tr>
    <tr>
      <th>feat_c</th>
      <td>0.381435</td>
      <td>0.422564</td>
      <td>0.635896</td>
      <td>0.445776</td>
    </tr>
    <tr>
      <th>feat_d</th>
      <td>0.234878</td>
      <td>0.334312</td>
      <td>0.445776</td>
      <td>0.349302</td>
    </tr>
  </tbody>
</table>
</div>


![Have you ever questioned the nature of your reality?](assets/doesnt_look_like_anything.jpg)

Its covariance matrix $C_X$ doesn't look that intersting either. However, let us recall that the covariance matrix is always a symmetric matrix with the variances on its diagonal and the covariances off-diagonal, i.e., having the following form:

$$
\large
\begin{vmatrix}
var(a, a) &  cov(a, b) &  cov(a, c) \\
cov(b, a) &  var(b, b) &  cov(b, c) \\
cov(c, a) &  cov(c, b) &  var(c, c) \\
\end{vmatrix}
$$

Let's also recall that $cov(x, y)$ is zero if and only if feature x and y are uncorrelated. The non-zero convariances in $C_X$ is an indication that there are quite some redundant feature measurements in $X$. So what we are trying to do here is feature extraction. We would like to rotate our dataset in a way such that the change of basis will represented by features that are uncorrelated to each other, i.e., having a new covariance matrix that is diagonal.

### Time to Choose
With a clearer goal now, let's figure out how we can achieve it.

$$
\begin{array}{ccc}
\text{Givens} & \text{Goal} & \text{Unknown} \\
\hline
\begin{gathered}Y = PX \\ C_X \equiv \frac{1}{n}XX^T \\ C_Y \equiv \frac{1}{n}YY^T\end{gathered}
& C_Y\text{ to be diagonal}
& \text{How to Choose }P\text{?}
\end{array}
$$

From the givens above, we are able to derive the relationship between $C_Y$ and $C_X$ in terms of $P$:

$$
C_Y = \frac{1}{n}YY^T = \frac{1}{n}(PX)(PX)^T = \frac{1}{n}PXX^TP^T
$$

$$
C_Y = PC_XP^T
$$

Let's recall one more time that all covariance matrices are symmetric, and any symmetric matrix can be "Eigendecomposed" as

$$
Q{\Lambda}Q^T
$$

where $Q$ is an orthogonal matrix whose columns are the eigenvectors of the symmetric matrix, and $\Lambda$ is a diagonal matrix whose entries are the eigenvalues. There is usally more than one way to choose $P$, but Eigendecomposing $C_X$ will prove to make our life much easier. Let's see what we can do with it:

$$
C_Y = PQ{\Lambda}Q^TP^T
$$

Since we know $\Lambda$ is diagonal and $Q^TQ \equiv I$, what if we choose $P$ to be $Q^T$?

$$
C_Y = Q^TQ{\Lambda}Q^TQ = I{\Lambda}I
$$

$$
C_Y = \Lambda
$$

Voilà, by choosing $P$ to eigenvectors of $C_X$, we are able to transform $X$ into $Y$ whose features are uncorrelated to each other!

### Show
Well, that was quite convenient, wasn't it? What's even better is that we can demonstrate it in a few lines of code:


```python
_, q = np.linalg.eig(c_x)  # Eigendecomposition
p = q.transpose()
y = tabulate(p @ x, rotated=True)
display_df(y)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample0</th>
      <th>sample1</th>
      <th>sample2</th>
      <th>sample3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>new_feat_e</th>
      <td>1.400186</td>
      <td>1.576029</td>
      <td>0.906121</td>
      <td>0.830166</td>
    </tr>
    <tr>
      <th>new_feat_f</th>
      <td>-0.162144</td>
      <td>-0.225848</td>
      <td>0.572904</td>
      <td>0.076917</td>
    </tr>
    <tr>
      <th>new_feat_g</th>
      <td>-0.042285</td>
      <td>-0.086877</td>
      <td>-0.091316</td>
      <td>0.335921</td>
    </tr>
    <tr>
      <th>new_feat_h</th>
      <td>0.087761</td>
      <td>-0.072164</td>
      <td>-0.002497</td>
      <td>-0.008295</td>
    </tr>
  </tbody>
</table>
</div>


The transformed dataset $Y$ with the newly extracted features e to h doesn't look like anything either, but what about its convariance matrix??


```python
c_y = covariance_matrix(y)
display_df(c_y)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>new_feat_e</th>
      <th>new_feat_f</th>
      <th>new_feat_g</th>
      <th>new_feat_h</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>new_feat_e</th>
      <td>1.488654</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>new_feat_f</th>
      <td>0.000000</td>
      <td>0.102858</td>
      <td>0.000000</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>new_feat_g</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032629</td>
      <td>-0.000000</td>
    </tr>
    <tr>
      <th>new_feat_h</th>
      <td>0.000000</td>
      <td>-0.000000</td>
      <td>-0.000000</td>
      <td>0.003246</td>
    </tr>
  </tbody>
</table>
</div>


Holy moly, isn't this exactly what we were aiming for, with just a few lines of code? From a dataset with some redundant and less interesting fetures, we have extracted new features that are much more meaningful to look at, simply by diagonalizing its convariance matrix. Let's wrap this up with some side-by-side comparisons.


```python
display_df(x, latex=True)
display_df(c_x, latex=True)
display_df(y, latex=True)
display_df(c_y, latex=True)
```


$$
\footnotesize
\begin{array} {cc}
X & \text{Covariance Matrix of }X \\
\begin{array}{|l|rrrr|}
\hline
{} &   sample0 &   sample1 &   sample2 &   sample3 \\
\hline
feat\_a &  0.472612 &  0.453242 &  0.811147 &  0.237625 \\
feat\_b &  0.728994 &  0.916212 &  0.202783 &  0.116406 \\
feat\_c &  0.803590 &  0.967202 &  0.659594 &  0.726142 \\
feat\_d &  0.771849 &  0.753178 &  0.153215 &  0.459026 \\
\hline
\end{array}
&
\begin{array}{|l|rrrr|}
\hline
{} &    feat\_a &    feat\_b &    feat\_c &    feat\_d \\
\hline
feat\_a &  0.285804 &  0.237986 &  0.381435 &  0.234878 \\
feat\_b &  0.237986 &  0.356387 &  0.422564 &  0.334312 \\
feat\_c &  0.381435 &  0.422564 &  0.635896 &  0.445776 \\
feat\_d &  0.234878 &  0.334312 &  0.445776 &  0.349302 \\
\hline
\end{array}
\\
\end{array}
$$


$$
\footnotesize
\begin{array} {cc}
Y & \text{Covariance Matrix of }Y \\
\begin{array}{|l|rrrr|}
\hline
{} &   sample0 &   sample1 &   sample2 &   sample3 \\
\hline
new\_feat\_e &  1.400186 &  1.576029 &  0.906121 &  0.830166 \\
new\_feat\_f & -0.162144 & -0.225848 &  0.572904 &  0.076917 \\
new\_feat\_g & -0.042285 & -0.086877 & -0.091316 &  0.335921 \\
new\_feat\_h &  0.087761 & -0.072164 & -0.002497 & -0.008295 \\
\hline
\end{array}
&
\begin{array}{|l|rrrr|}
\hline
{} &  new\_feat\_e &  new\_feat\_f &  new\_feat\_g &  new\_feat\_h \\
\hline
new\_feat\_e &    1.488654 &    0.000000 &    0.000000 &    0.000000 \\
new\_feat\_f &    0.000000 &    0.102858 &    0.000000 &   -0.000000 \\
new\_feat\_g &    0.000000 &    0.000000 &    0.032629 &   -0.000000 \\
new\_feat\_h &    0.000000 &   -0.000000 &   -0.000000 &    0.003246 \\
\hline
\end{array}
\\
\end{array}
$$


Look at this. Isn't it just beautiful?

[1]: The reason orthonormality is part of the goal is that we do not want to do anything more than rotations. We do not want to modify $X$. We only want to re-express $X$ by carefully choosing a change of basis.
