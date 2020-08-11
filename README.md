# ESTRNN
[Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring (ECCV2020 Spotlight)](https://github.com/zzh-tech/ESTRNN/)  

by Zhihang Zhong, Ye Gao, Yinqiang Zheng, Bo Zheng


## Results
### Results on REDS
![image](./imgs/reds.gif)


### Results on GOPRO
![image](./imgs/gopro.gif)


### Results on BSD
![image](./imgs/bsd.gif)


## Prerequisites
- Python 3.7
- PyTorch 1.4 with GPU
- opencv-python
- lmdb


## Training
Take GOPRO as an example, we first prepared the datasets in LMDB format.  

You can download the original down-sampled GOPRO dataset [("*gopro_ds*")](https://drive.google.com/file/d/1vZutfe4pjm9anDtdJPc1f3mu62pDtXt_/view?usp=sharing) and use "*./tool/lmdb_gopro_ds.ipynb*" to create "*gopro_ds_lmdb*", or directly download the one we have made  [("*gopro_ds_lmdb*")](https://drive.google.com/drive/folders/1oWn-noXnO5xpbud8nknmpITvBZ6PZoIE?usp=sharing).

Then, please specify the *\<path\>* (e.g. "*./dataset/* ") where you put the folder "*gopro_ds_lmdb*" in command, or change the default value of "*data_root*" in "*./para/\_\_init\_\_.py*".

Training command is as below:

```bash
python main.py --data_root <path>
```

You can also tune the hyper parameters such as batch size, learning rate, epoch number, etc., by specifying it in command or changing the corresponding value in "*./para/\_\_init\_\_.py*".   
```bash
python main.py --lr 1e-4 --batch_size 4 --num_gpus 2 --trainer_mode ddp
```


## Beam-Splitter Dataset (BSD)
Now, we are trying to collect a more complete beam-splitter dataset for video deblurring, using the proposed beam-splitter capture system as below:  

![image](./imgs/bsd_system.png)
![image](./imgs/bsd_demo.gif)


We will release our BSD dataset soon...

## Citing
If you use any part of our code, or ESTRNN and BSD are useful for your research, please consider citing:

```bibtex
@InProceedings{Zhong_2020_ECCV,
  title={Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring},
  author={Zhong, Zhihang and Ye, Gao and Zheng, Yinqiang and Bo, Zheng},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  month = {August},
  year={2020}
}
```

## Contact
We are glad to hear if you have any suggestions and questions.  

Please send email to *zzh-tech@gmail.com*
