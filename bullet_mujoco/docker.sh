# Install necessary package
apt-get -y update && apt-get -y install ffmpeg
apt-get -y install vim htop
apt-get -y --fix-broken install
apt-get -y install libglew-dev
# RUN apt-get -y install libglew2.0

apt-get -y update && apt-get -y install libglfw3 libgl1-mesa-glx libosmesa6 libglew-dev

# setting system
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin >> /home/.bashrc
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200_linux/bin  >> /home/.bashrc
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia  >> /home/.bashrc
echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin  >> /home/.bashrc
echo export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so >> /home/.bashrc
apt-get -y install libgl1-mesa-glx libosmesa6 libglew-dev
echo export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so >> /home/.bashrc
echo export MUJOCO_GL=egl >> /home/.bashrc

# Install rl package
pip install git+git://github.com/denisyarats/dmc2gym.git
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install pandas imageio imageio-ffmpeg scikit-image tb-nightly absl-py pyparsing
pip install gym termcolor
pip install pillow mujoco
pip install dm_control
pip install pynvml
pip install tensorboard
