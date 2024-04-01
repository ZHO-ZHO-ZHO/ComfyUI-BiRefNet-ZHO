![BRF](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO/assets/140084057/ad43b70b-370b-49ca-94df-46039a224ce5)


# ComfyUI-BiRefNet-ZHO

Better version for [BiRefNet](https://github.com/zhengpeng7/birefnet) in ComfyUI | Both img & video

![Dingtalk_20240401154248](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO/assets/140084057/1d72a017-5dc5-482e-a0c7-284d14b807b7)


## 项目介绍 | Info

- 对 [BiRefNet](https://github.com/zhengpeng7/birefnet) 的非官方实现

- 与 [viperyl/ComfyUI-BiRefNet](https://github.com/viperyl/ComfyUI-BiRefNet) 插件区别：
  
   - 原版插件：只能简单输出蒙版，不方便用，也不能处理视频
     
   - 新版插件：
     
      1）模型加载和图像处理相分离，提升速度（和我之前做的 [BRIA RMBG in ComfyUI](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BRIA_AI-RMBG) 插件一致）
 
      2）可以直接输出透明背景的 PNG 图
     
      3）可以直接抠视频

- BiRefNet 模型：目前最好的开源可商用背景抠除模型

- 版本：**V1.0** 同时支持 图像和视频 处理


## 视频演示


https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO/assets/140084057/ae337aa0-ec9e-40c3-85d4-390654bd0cc7


## 安装 | Install

- 所需依赖：timm，如已安装无需运行 requirements.txt，只需 git 项目即可

- 推荐使用管理器 ComfyUI Manager 安装（On the Way）

- 手动安装：
    1. `cd custom_nodes`
    2. `git clone https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO.git`
    3. `cd custom_nodes/ComfyUI-BiRefNet-ZHO`
    4. `pip install -r requirements.txt`
    5. 重启 ComfyUI


## 使用说明 | How to Use

- 将 [BiRefNet](https://huggingface.co/ViperYX/BiRefNet) 中的 6 个模型均下载至`./models/BiRefNet`

- 节点：

  ![Dingtalk_20240331031811](https://github.com/ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO/assets/140084057/ce37a57e-c5d0-4f17-9a87-457dd1022776)


   - 🧹BiRefNet Model Loader：自动加载 BiRefNet 模型
   - 🧹BiRefNet：去除背景


## 更新日志

- 20240401

  V1.0 同时支持 图像和视频 处理（支持批量处理）、支持输出 mask 功能

  创建项目 


## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO&type=Date)](https://star-history.com/#ZHO-ZHO-ZHO/ComfyUI-BiRefNet-ZHO&Date)


## Credits

[BiRefNet](https://github.com/zhengpeng7/birefnet)

代码参考了 [viperyl/ComfyUI-BiRefNet](https://github.com/viperyl/ComfyUI-BiRefNet) 感谢！
