# MHGTCDA
Prediction of circRNA-drug sensitivity using random auto-encoders and multi-layer heterogeneous graph Transformers

# Introduction

MHGTCDA is a cutting-edge model developed for predicting circRNA-drug sensitivity associations in various cellular contexts. This model employs a sophisticated strategy combining adaptive random auto-encoders (RAEs) and multilayer Heterogeneous Graph Transformers (MHGT) to enhance prediction accuracy across multiple species. By treating circRNAs and drug molecules as complex networks, MHGTCDA adaptively encodes these entities to capture essential latent representations. The MHGT framework further processes these encodings by integrating contextual node representations with edge information from bipartite graphs of circRNA-drug pairs, minimizing potential information loss.

For each target condition, MHGTCDA was fine-tuned with specific training datasets, allowing for customized predictions tailored to the unique biological characteristics of each drug-circRNA interaction. This innovative approach not only improves the reliability of predictions but also provides deeper insights into the molecular interactions influencing drug efficacy.

Extensive cross-validation tests against nine other advanced models and three conventional methods have shown that MHGTCDA significantly surpasses existing techniques in terms of predictive performance. Detailed case studies on three specific diseases further validate its effectiveness and utility in real-world scenarios.

For a comprehensive understanding of the MHGTCDA model and its applications in circRNA-drug sensitivity prediction, please refer to our detailed publication, Prediction of circRNA-drug sensitivity using random auto-encoders and multi-layer heterogeneous graph Transformers.

# Requirements
