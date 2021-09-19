FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y libgtk-3-dev libnotify-dev libsdl2-dev python3-pip
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && python3 -m pip install https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-20.04/wxPython-4.1.1-cp38-cp38-linux_x86_64.whl && python3 -m pip install --no-cache-dir -r requirements.txt
