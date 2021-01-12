# SFSeg

* **Title**: A 3D Convolutional Approach to Spectral Object Segmentation in Space and Time
* **Authors**: Elena Burceanu, Marius Leordeanu
* **Paper** (published at IJCAI-2020): https://www.ijcai.org/Proceedings/2020/69

<p float="left">
  <img src="resources/intuition_1.png" width="540" />
  <img src="resources/intuition_2.png" width="300" /> 
</p>



### Key aspects:
1. **Formulating segmentation in video** as a problem of finding the **main space-time cluster**, represented by the leading eigenvector of the pixel-level adjacency matrix of the **video's graph in space-time**.
2. **Fast algorithm: SFSeg** is a **3D spectral filtering algorithm**, that computes the main eigenvector **without explicitly computing the graph’s adjacency matrix**. This transforms the problem into a **tractable** one. 
3. **Refinement: SFSeg** can be used as a powerful refinement method. It is **faster and more accurate** then the well known space-time approach using CRF (denseCRF).


### Run:
- input: input_masks.th (unary term) and features.th (pairwise term)
- output: output segmentation
- sample folder: input_masks.th and features.th maps are a segmentation map (the same)
`python main.py`


### Please refer to it as:
```
@inproceedings{burceanu-sfseg,
  title     = {A 3D Convolutional Approach to Spectral Object Segmentation in Space and Time},
  author    = {Burceanu, Elena and Leordeanu, Marius},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  url       = {https://doi.org/10.24963/ijcai.2020/69},
}
```
