# Setup WhisperTRT

# NOTE

Jetson Containers doesn't work unless on Jetson Orin nano, this is here for version control.

## install Jetson Containers
```bash
# Clone the repository
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers

# Run the installer
bash install.sh

# Add to PATH (restart terminal after this or source your bashrc)
echo 'export PATH=$PATH:~/jetson-containers' >> ~/.bashrc
source ~/.bashrc
```

## Test faster-whisper container
```bash
# Pull/build and run faster-whisper container
jetson-containers run $(autotag faster-whisper)

# This will drop you into a container shell
# Test if it works:
python3 -c "from faster_whisper import WhisperModel; print('faster-whisper loaded successfully!')"
```

## Make run_docker executable

```bash
chmod +x run_docker.sh
```

## Run it

```bash
./run_docker.sh
```