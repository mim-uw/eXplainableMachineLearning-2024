# Homework 3

# 3. Decompositions using LIME

I have ran decompositions, on linerar model and DNN

From observations on few samples, I think that linear model explanations are much more chaotic wheras, for DNN, features like Alptha, Length, Width, Dist, M3Long were consistently important.

I also have ran LIME using larger number of samples (1000), with a goal to try to establish some larger patterns. 

I used explanation.to_map(), but I might be missusing this function.

To see those graphs, look into the attached rendered jupyter notebook.

# 4. Decompositions using SHAP

First big observation is that SHAP is much slower than LIME. I think it might have quadratic complexity, so I have ran it only on 100 samples.

Due to how my code is structured, SHAP explanations are sorted
