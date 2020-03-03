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
> The [original] dataset is $X$, an $m × n$ matrix.<br>
> Find some orthonormal matrix $P$ in $Y = PX$ such that $C_Y \equiv \frac{1}{n}YY^T$ is a diagonal matrix.[1]<br>
> The rows of $P$ shall be principal components of $X$.

As you might have noticed, $C_Y$ here is the covariance matrix of our rotated dataset $Y$. Why do we want $C_Y$ to be diagonal? Before we answer this question, let’s generate a dataset $X$ consisting of 4 features with some random values.


```python
import numpy as np
feature_num, sample_num = 4, 4
x = np.random.rand(feature_num, sample_num)

def covariance_matrix(dataset):
    sample_count = dataset.shape[1]
    return (dataset @ dataset.transpose() / sample_count)
```


```python
c_x = covariance_matrix(x)
_, p = np.linalg.eig(c_x)
y = p.transpose() @ x
c_y = covariance_matrix(y)
```


```python
from IPython.display import HTML, Latex, display
import pandas as pd

def display_side_by_side(df1, df2, name1, name2):
    inline = 'style="display: float; max-width:50%" class="table"'
    q = '''
    <div class="table-responsive col-md-6">{}</div>
    <div class="table-responsive col-md-6">{}</div>
    '''.format(df1.style.set_table_attributes(inline).set_caption(name1).render(),
               df2.style.set_table_attributes(inline).set_caption(name2).render())
    return HTML(q)

df_x = pd.DataFrame.from_records(x, columns=['sample{}'.format(num) for num in range(sample_num)], index=['feat_{}'.format(name) for name in 'abcd']).round(15)
df_cx = pd.DataFrame.from_records(c_x, columns=['feat__{}'.format(name) for name in 'abcd'], index=['feat_{}'.format(name) for name in 'abcd']).round(15)
df_y = pd.DataFrame.from_records(y, columns=['sample{}'.format(num) for num in range(sample_num)], index=['new_feat_{}'.format(name) for name in 'efgh']).round(15)
df_cy = pd.DataFrame.from_records(c_y, columns=['new_feat__{}'.format(name) for name in 'efgh'], index=['new_feat_{}'.format(name) for name in 'efgh']).round(15)
display(Latex(df_y.to_latex()))
print(df_y.to_latex())
display(Latex(df_cy.to_latex()))
print(df_cy.to_latex())
# display(df_x)
# display(df_cx)
# display(df_y)
# display(df_cy)

# display_side_by_side(df_x, df_cx, 'X', 'Covariance Matrix of X')
# display_side_by_side(df_y, df_cy, 'Y', 'Covariance Matrix of Y')
# display(df_x.style.set_caption('X'))
# display(df_cx.style.set_caption('Covariance Matrix of X'))
# display(df_y.style.set_caption('Y'))
# display(df_cy.style.set_caption('Covariance Matrix of Y'))
```


$$
\small
\begin{array} {cc}
X & \text{Covariance Matrix of }X \\
\begin{array}{|l|rrrr|}
\hline
{} &   sample0 &   sample1 &   sample2 &   sample3 \\
\hline
new\_feat\_e &  1.237100 &  1.163050 &  1.130956 &  0.699606 \\
new\_feat\_f & -0.430329 & -0.011550 &  0.560685 & -0.126239 \\
new\_feat\_g &  0.098891 &  0.074781 &  0.007388 & -0.311131 \\
new\_feat\_h &  0.048906 & -0.083760 &  0.034965 & -0.003757 \\
\hline
\end{array}
&
\begin{array}{|l|rrrr|}
\hline
{} &  new\_feat\_\_e &  new\_feat\_\_f &  new\_feat\_\_g &  new\_feat\_\_h \\
\hline
new\_feat\_e &     1.162903 &     0.000000 &     0.000000 &     0.000000 \\
new\_feat\_f &     0.000000 &     0.128905 &     0.000000 &    -0.000000 \\
new\_feat\_g &     0.000000 &     0.000000 &     0.028057 &    -0.000000 \\
new\_feat\_h &     0.000000 &    -0.000000 &    -0.000000 &     0.002661 \\
\hline
\end{array}
\\
\end{array}
$$
$$
\footnotesize
\begin{array} {cc}
X & \text{Covariance Matrix of }X \\
\begin{array}{|l|rrrr|}
\hline
{} &   sample0 &   sample1 &   sample2 &   sample3 \\
\hline
new\_feat\_e &  1.237100 &  1.163050 &  1.130956 &  0.699606 \\
new\_feat\_f & -0.430329 & -0.011550 &  0.560685 & -0.126239 \\
new\_feat\_g &  0.098891 &  0.074781 &  0.007388 & -0.311131 \\
new\_feat\_h &  0.048906 & -0.083760 &  0.034965 & -0.003757 \\
\hline
\end{array}
&
\begin{array}{|l|rrrr|}
\hline
{} &  new\_feat\_\_e &  new\_feat\_\_f &  new\_feat\_\_g &  new\_feat\_\_h \\
\hline
new\_feat\_e &     1.162903 &     0.000000 &     0.000000 &     0.000000 \\
new\_feat\_f &     0.000000 &     0.128905 &     0.000000 &    -0.000000 \\
new\_feat\_g &     0.000000 &     0.000000 &     0.028057 &    -0.000000 \\
new\_feat\_h &     0.000000 &    -0.000000 &    -0.000000 &     0.002661 \\
\hline
\end{array}
\\
\end{array}
$$
$$
\scriptsize
\begin{array} {cc}
X & \text{Covariance Matrix of }X \\
\begin{array}{|l|rrrr|}
\hline
{} &   sample0 &   sample1 &   sample2 &   sample3 \\
\hline
new\_feat\_e &  1.237100 &  1.163050 &  1.130956 &  0.699606 \\
new\_feat\_f & -0.430329 & -0.011550 &  0.560685 & -0.126239 \\
new\_feat\_g &  0.098891 &  0.074781 &  0.007388 & -0.311131 \\
new\_feat\_h &  0.048906 & -0.083760 &  0.034965 & -0.003757 \\
\hline
\end{array}
&
\begin{array}{|l|rrrr|}
\hline
{} &  new\_feat\_\_e &  new\_feat\_\_f &  new\_feat\_\_g &  new\_feat\_\_h \\
\hline
new\_feat\_e &     1.162903 &     0.000000 &     0.000000 &     0.000000 \\
new\_feat\_f &     0.000000 &     0.128905 &     0.000000 &    -0.000000 \\
new\_feat\_g &     0.000000 &     0.000000 &     0.028057 &    -0.000000 \\
new\_feat\_h &     0.000000 &    -0.000000 &    -0.000000 &     0.002661 \\
\hline
\end{array}
\\
\end{array}
$$
$$
\tiny
\begin{array} {cc}
X & \text{Covariance Matrix of }X \\
\begin{array}{|l|rrrr|}
\hline
{} &   sample0 &   sample1 &   sample2 &   sample3 \\
\hline
new\_feat\_e &  1.237100 &  1.163050 &  1.130956 &  0.699606 \\
new\_feat\_f & -0.430329 & -0.011550 &  0.560685 & -0.126239 \\
new\_feat\_g &  0.098891 &  0.074781 &  0.007388 & -0.311131 \\
new\_feat\_h &  0.048906 & -0.083760 &  0.034965 & -0.003757 \\
\hline
\end{array}
&
\begin{array}{|l|rrrr|}
\hline
{} &  new\_feat\_\_e &  new\_feat\_\_f &  new\_feat\_\_g &  new\_feat\_\_h \\
\hline
new\_feat\_e &     1.162903 &     0.000000 &     0.000000 &     0.000000 \\
new\_feat\_f &     0.000000 &     0.128905 &     0.000000 &    -0.000000 \\
new\_feat\_g &     0.000000 &     0.000000 &     0.028057 &    -0.000000 \\
new\_feat\_h &     0.000000 &    -0.000000 &    -0.000000 &     0.002661 \\
\hline
\end{array}
\\
\end{array}
$$


[1]: The reason orthonormality is part of the goal is that we do not want to do anything more than rotating $X$. We do not want to modify $X$. We only want to re-express $X$ by carefully choosing a change of basis.
