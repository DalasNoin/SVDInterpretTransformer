# SVDInterpretTransformer

Apply SVD to Transformer weights

Based on a Conjecture Publication

[1] https://www.alignmentforum.org/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight

[2] https://github.com/BerenMillidge/svd_directions

# Usage

Examples are provided in svd_directions/examples.py

## Todo

- [ ] TopKTable should show information about the layer and head
- [ ] maybe make it possible to show different embeddings at the same time
- [ ] show a plot of the correlation between singular values and the cosine similarity of the embeddings(action, state, return, ...)