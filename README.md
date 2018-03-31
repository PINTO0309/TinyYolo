# [Japanese] TinyYolo
YoloV2+Neural Compute Stick(NCS)+Raspberry Piの限界性能に挑戦　Challenge the marginal performance of YoloV2 + Neural Compute Stick + Raspberry Pi

https://qiita.com/PINTO/items/db3ab44a3e2bcd87f2d8

# 動作イメージ
TinyYolo + Neural Compute Stick + RaspberryPi3

![Riders](https://github.com/PINTO0309/MobileNet-SSD/blob/master/media/Riders.gif)  ![MultiStick](https://github.com/PINTO0309/MobileNet-SSD/blob/master/media/MultiStick.jpeg)
# 環境
・RaspberryPi 3 + Raspbian Stretch

・NCSDK v1.12.00

・Intel Movidius Neural Compute Stick　１本

・OpenCV 3.4.1

・OpenGL

・numpy

・UVC対応のUSB-Webカメラ


# 環境構築
1. パッケージのインストール
```
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install python3-pip python3-numpy git cmake
```
2. NCSDKのインストール
```
$ cd ~
$ git clone https://github.com/movidius/ncsdk.git
$ cd ncsdk
$ make install
```
3. OpenCVのインストール
```
$ wget https://github.com/PINTO0309/OpenCVonARMv7/blob/master/libopencv3_3.4.1-20180304.1_armhf.deb
$ sudo apt install -y ./libopencv3_3.4.1-20180304.1_armhf.deb
$ sudo ldconfig
```
4. OpenGLのインストール
```
$ sudo apt-get install python-opengl
$ sudo -H pip3 install pyopengl
$ sudo -H pip3 install pyopengl_accelerate
$ sudo raspi-config
```
5. 「7.Advanced Options」-「A7 GL Driver」-「G2 GL (Fake KMS)」の順に選択し、Raspberry Pi のOpenGL Driver を有効化

6. 再起動
```
$ sudo reboot
```
7. リソース一式のダウンロード
```
$ cd ~
$ git clone https://github.com/PINTO0309/MobileNet-SSD.git
```
8. USB-WEBカメラ(UVC対応) と Neural Compute Stick をRaspberryPiのUSBポートへ接続(Neural Compute Stickをマルチで使用する場合は電圧が不足するためセルフパワーUSB-Hub必須)

9. RaspberryPiとディスプレイをHDMIケーブルで接続

10. MobileNet-SSDの実行
```
$ cd MobileNet-SSD
$ python3 MultiStickSSD.py
```

　
　
 
# [English] MobileNet-SSD
Ultra-fast MobileNet-SSD + Neural Compute Stick(NCS) than YoloV2 + Explosion speed by RaspberryPi · Multiple moving object detection with high accuracy

https://qiita.com/PINTO/items/b97b3334ed452cb555e2

# Image of motion
MobileNet-SSD + Neural Compute Stick + RaspberryPi3 / MultiStick(3 Stick / Hard Motion)

![Riders](https://github.com/PINTO0309/MobileNet-SSD/blob/master/media/Riders.gif)  ![MultiStick](https://github.com/PINTO0309/MobileNet-SSD/blob/master/media/MultiStick.jpeg)
# Environment
・RaspberryPi 3 + Raspbian Stretch

・NCSDK v1.12.00

・Intel Movidius Neural Compute Stick　１本

・OpenCV 3.4.1

・OpenGL

・numpy

・(UVC)USB-Web Camera


# Building environment
1. Installing packages
```
$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install python3-pip python3-numpy git cmake
```
2. Installing NCSDK
```
$ cd ~
$ git clone https://github.com/movidius/ncsdk.git
$ cd ncsdk
$ make install
```
3. Installation of OpenCV
```
$ wget https://github.com/PINTO0309/OpenCVonARMv7/blob/master/libopencv3_3.4.1-20180304.1_armhf.deb
$ sudo apt install -y ./libopencv3_3.4.1-20180304.1_armhf.deb
$ sudo ldconfig
```
4. Installing OpenGL
```
$ sudo apt-get install python-opengl
$ sudo -H pip3 install pyopengl
$ sudo -H pip3 install pyopengl_accelerate
$ sudo raspi-config
```
5. 「7.Advanced Options」-「A7 GL Driver」-「G2 GL (Fake KMS)」 and Activate Raspberry Pi's OpenGL Driver

6. Reboot
```
$ sudo reboot
```
7. Download complete set of resources
```
$ cd ~
$ git clone https://github.com/PINTO0309/MobileNet-SSD.git
```
8. Connect USB-WEB camera (UVC compatible) and Neural Compute Stick to RaspberryPi's USB port (self power USB-Hub required due to insufficient voltage when using Neural Compute Stick in multiple)

9. Connect RaspberryPi and display with HDMI cable

10. Running MobileNet-SSD
```
$ cd MobileNet-SSD
$ python3 MultiStickSSD.py
```
