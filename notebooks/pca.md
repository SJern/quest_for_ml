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
> The [original] dataset is $$X$$, an $$m × n$$ matrix.<br>
> Find some orthonormal matrix $$P$$ in $$Y = PX$$ such that $$C_Y \equiv \frac{1}{n}YY^T$$ is a diagonal matrix.[1]<br>
> The rows of $$P$$ shall be principal components of $$X$$.

As you might have noticed, $$C_Y$$ here is the covariance matrix of our rotated dataset $$Y$$. Why do we want $$C_Y$$ to be diagonal? Before we answer this question, let’s generate a dataset $$X$$ consisting of 4 features with some random values.


```python
import numpy as np
feature_num, sample_num = 4, 6
x = np.random.rand(feature_num, sample_num)

def covariance_matrix(dataset):
    sample_count = dataset.shape[1]
    return (dataset @ dataset.transpose() / sample_count).round(15)
```


```python
c_x = covariance_matrix(x)
_, p = np.linalg.eig(c_x)
y = p.transpose() @ x
c_y = covariance_matrix(y)
```


```python
from IPython.display import HTML, display
import pandas as pd

def display_side_by_side(df1, df2, name1, name2):
    inline = 'style="display: float; max-width:50%" class="table"'
    q = '''
    <div class="table-responsive col-md-6">{}</div>
    <div class="table-responsive col-md-6">{}</div>
    '''.format(df1.style.set_table_attributes(inline).set_caption(name1).render(),
               df2.style.set_table_attributes(inline).set_caption(name2).render())
    return HTML(q)

df_x = pd.DataFrame.from_records(x, columns=['sample{}'.format(num) for num in range(sample_num)], index=['feat_{}'.format(name) for name in 'abcd'])
df_cx = pd.DataFrame.from_records(c_x, columns=['feat__{}'.format(name) for name in 'abcd'], index=['feat_{}'.format(name) for name in 'abcd'])
df_y = pd.DataFrame.from_records(y, columns=['sample{}'.format(num) for num in range(sample_num)], index=['new_feat_{}'.format(name) for name in 'efgh'])
df_cy = pd.DataFrame.from_records(c_y, columns=['new_feat__{}'.format(name) for name in 'efgh'], index=['new_feat_{}'.format(name) for name in 'efgh'])

# display_side_by_side(df_x, df_cx, 'X', 'Covariance Matrix of X')
# display_side_by_side(df_y, df_cy, 'Y', 'Covariance Matrix of Y')
display(df_x.style.set_caption('X'))
display(df_cx.style.set_caption('Covariance Matrix of X'))
display(df_y.style.set_caption('Y'))
display(df_cy.style.set_caption('Covariance Matrix of Y'))
```

<div style="display: flex; flex-direction: row; justify-content: space-evenly;">
<style  type="text/css" >
</style><table id="T_ff170538_5b46_11ea_b65b_acbc32c2f94f" ><caption style="text-align: center;">X</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >sample0</th>        <th class="col_heading level0 col1" >sample1</th>        <th class="col_heading level0 col2" >sample2</th>        <th class="col_heading level0 col3" >sample3</th>        <th class="col_heading level0 col4" >sample4</th>        <th class="col_heading level0 col5" >sample5</th>    </tr></thead>
<tbody>
                <tr>
                        <th id="T_ff170538_5b46_11ea_b65b_acbc32c2f94flevel0_row0" class="row_heading level0 row0" >feat_a</th>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow0_col0" class="data row0 col0" >0.293459</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow0_col1" class="data row0 col1" >0.077714</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow0_col2" class="data row0 col2" >0.905751</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow0_col3" class="data row0 col3" >0.613897</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow0_col4" class="data row0 col4" >0.060830</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow0_col5" class="data row0 col5" >0.379016</td>
            </tr>
            <tr>
                        <th id="T_ff170538_5b46_11ea_b65b_acbc32c2f94flevel0_row1" class="row_heading level0 row1" >feat_b</th>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow1_col0" class="data row1 col0" >0.378693</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow1_col1" class="data row1 col1" >0.113846</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow1_col2" class="data row1 col2" >0.214324</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow1_col3" class="data row1 col3" >0.538663</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow1_col4" class="data row1 col4" >0.691506</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow1_col5" class="data row1 col5" >0.941329</td>
            </tr>
            <tr>
                        <th id="T_ff170538_5b46_11ea_b65b_acbc32c2f94flevel0_row2" class="row_heading level0 row2" >feat_c</th>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow2_col0" class="data row2 col0" >0.080650</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow2_col1" class="data row2 col1" >0.274110</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow2_col2" class="data row2 col2" >0.981330</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow2_col3" class="data row2 col3" >0.051426</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow2_col4" class="data row2 col4" >0.970625</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow2_col5" class="data row2 col5" >0.987066</td>
            </tr>
            <tr>
                        <th id="T_ff170538_5b46_11ea_b65b_acbc32c2f94flevel0_row3" class="row_heading level0 row3" >feat_d</th>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow3_col0" class="data row3 col0" >0.571006</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow3_col1" class="data row3 col1" >0.336628</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow3_col2" class="data row3 col2" >0.905613</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow3_col3" class="data row3 col3" >0.825542</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow3_col4" class="data row3 col4" >0.441280</td>
                        <td id="T_ff170538_5b46_11ea_b65b_acbc32c2f94frow3_col5" class="data row3 col5" >0.318327</td>
            </tr>
    </tbody></table>



<style  type="text/css" >
</style><table id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94f" ><caption style="text-align: center;">Covariance Matrix of X</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >feat__a</th>        <th class="col_heading level0 col1" >feat__b</th>        <th class="col_heading level0 col2" >feat__c</th>        <th class="col_heading level0 col3" >feat__d</th>    </tr></thead>
<tbody>
                <tr>
                        <th id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94flevel0_row0" class="row_heading level0 row0" >feat_a</th>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow0_col0" class="data row0 col0" >0.239461</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow0_col1" class="data row0 col1" >0.173938</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow0_col2" class="data row0 col2" >0.233090</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow0_col3" class="data row0 col3" >0.278047</td>
            </tr>
            <tr>
                        <th id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94flevel0_row1" class="row_heading level0 row1" >feat_b</th>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow1_col0" class="data row1 col0" >0.173938</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow1_col1" class="data row1 col1" >0.309457</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow1_col2" class="data row1 col2" >0.316686</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow1_col3" class="data row1 col3" >0.249690</td>
            </tr>
            <tr>
                        <th id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94flevel0_row2" class="row_heading level0 row2" >feat_c</th>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow2_col0" class="data row2 col0" >0.233090</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow2_col1" class="data row2 col1" >0.316686</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow2_col2" class="data row2 col2" >0.493951</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow2_col3" class="data row2 col3" >0.302002</td>
            </tr>
            <tr>
                        <th id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94flevel0_row3" class="row_heading level0 row3" >feat_d</th>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow3_col0" class="data row3 col0" >0.278047</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow3_col1" class="data row3 col1" >0.249690</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow3_col2" class="data row3 col2" >0.302002</td>
                        <td id="T_ff18e9d4_5b46_11ea_b65b_acbc32c2f94frow3_col3" class="data row3 col3" >0.372847</td>
            </tr>
    </tbody></table>
</div>

<div style="display: flex; flex-direction: row; justify-content: space-evenly;">
<style  type="text/css" >
</style><table id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94f" ><caption style="text-align: center;">Y</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >sample0</th>        <th class="col_heading level0 col1" >sample1</th>        <th class="col_heading level0 col2" >sample2</th>        <th class="col_heading level0 col3" >sample3</th>        <th class="col_heading level0 col4" >sample4</th>        <th class="col_heading level0 col5" >sample5</th>    </tr></thead>
<tbody>
                <tr>
                        <th id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94flevel0_row0" class="row_heading level0 row0" >new_feat_e</th>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow0_col0" class="data row0 col0" >0.636530</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow0_col1" class="data row0 col1" >0.422693</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow0_col2" class="data row0 col2" >1.518393</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow0_col3" class="data row0 col3" >0.952671</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow0_col4" class="data row0 col4" >1.153582</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow0_col5" class="data row0 col5" >1.341163</td>
            </tr>
            <tr>
                        <th id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94flevel0_row1" class="row_heading level0 row1" >new_feat_f</th>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow1_col0" class="data row1 col0" >0.301831</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow1_col1" class="data row1 col1" >0.032134</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow1_col2" class="data row1 col2" >0.325724</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow1_col3" class="data row1 col3" >0.572897</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow1_col4" class="data row1 col4" >-0.502309</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow1_col5" class="data row1 col5" >-0.497041</td>
            </tr>
            <tr>
                        <th id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94flevel0_row2" class="row_heading level0 row2" >new_feat_g</th>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow2_col0" class="data row2 col0" >-0.078864</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow2_col1" class="data row2 col1" >-0.160210</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow2_col2" class="data row2 col2" >0.036548</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow2_col3" class="data row2 col3" >0.028764</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow2_col4" class="data row2 col4" >-0.181150</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow2_col5" class="data row2 col5" >0.181927</td>
            </tr>
            <tr>
                        <th id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94flevel0_row3" class="row_heading level0 row3" >new_feat_h</th>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow3_col0" class="data row3 col0" >-0.244105</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow3_col1" class="data row3 col1" >0.045671</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow3_col2" class="data row3 col2" >0.486327</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow3_col3" class="data row3 col3" >-0.338482</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow3_col4" class="data row3 col4" >-0.053299</td>
                        <td id="T_ff1988bc_5b46_11ea_b65b_acbc32c2f94frow3_col5" class="data row3 col5" >-0.162854</td>
            </tr>
    </tbody></table>



<style  type="text/css" >
</style><table id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94f" ><caption style="text-align: center;">Covariance Matrix of Y</caption><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >new_feat__e</th>        <th class="col_heading level0 col1" >new_feat__f</th>        <th class="col_heading level0 col2" >new_feat__g</th>        <th class="col_heading level0 col3" >new_feat__h</th>    </tr></thead>
<tbody>
                <tr>
                        <th id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94flevel0_row0" class="row_heading level0 row0" >new_feat_e</th>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow0_col0" class="data row0 col0" >1.154401</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow0_col1" class="data row0 col1" >-0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow0_col2" class="data row0 col2" >-0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow0_col3" class="data row0 col3" >0.000000</td>
            </tr>
            <tr>
                        <th id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94flevel0_row1" class="row_heading level0 row1" >new_feat_f</th>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow1_col0" class="data row1 col0" >-0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow1_col1" class="data row1 col1" >0.170968</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow1_col2" class="data row1 col2" >0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow1_col3" class="data row1 col3" >0.000000</td>
            </tr>
            <tr>
                        <th id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94flevel0_row2" class="row_heading level0 row2" >new_feat_g</th>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow2_col0" class="data row2 col0" >-0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow2_col1" class="data row2 col1" >0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow2_col2" class="data row2 col2" >0.016660</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow2_col3" class="data row2 col3" >-0.000000</td>
            </tr>
            <tr>
                        <th id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94flevel0_row3" class="row_heading level0 row3" >new_feat_h</th>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow3_col0" class="data row3 col0" >0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow3_col1" class="data row3 col1" >0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow3_col2" class="data row3 col2" >-0.000000</td>
                        <td id="T_ff1a2e5c_5b46_11ea_b65b_acbc32c2f94frow3_col3" class="data row3 col3" >0.073687</td>
            </tr>
    </tbody></table>
</div>

[1]: The reason orthonormality is part of the goal is that we do not want to do anything more than rotating $$X$$. We do not want to modify $$X$$. We only want to re-express $$X$$ by carefully choosing a change of basis.
