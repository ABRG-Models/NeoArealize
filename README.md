# NeoArealize

A model of reaction-diffusion pattern formation in neocortex.

Pre-requisites:

Build and install morphologica from:

https://github.com/ABRG-Models/morphologica

Make sure these packages are installed (Debian/Ubuntu example):

sudo apt install python python-numpy xterm

Build and install jsoncpp (in a directory '~/src', just for example,
you can build it wherever suits):

```bash
mkdir -p ~/src
cd ~/src
git clone https://github.com/open-source-parsers/jsoncpp.git
cd jsoncpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=YES ..
make
sudo make install
```

Now you can build NeoArealize:

```bash
cd NeoArealize
mkdir build
cd build
cmake ..
make
cd ..
```

To run:

```bash
./build/sim/james1 ./config/c1.json
```

The program reads parameters from c1.json and writes results into
./logs/
