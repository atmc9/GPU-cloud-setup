> gcloud compute ssh gpu-deep-learner --zone us-east1-d

> source activate myenv
> conda create -n deepLearning --file environment.yml

Simple steps to setting up an GPU instance on Google-Cloud and running the first DeepLearning Model.

Are you super excited ??

Lets dive in

**Step 1: Install gcloud tools on your local machine.**
Follow either the
[google documentation link](https://cloud.google.com/sdk/downloads) or  [Installing Google Cloud SDK](https://www.youtube.com/watch?v=eGVc_WFzXtw)

Make sure you update the gcloud componenst if youa lready have it installed on your local, also install beta components

 > gcloud components update && gcloud components install beta

**Step 2:  Request for increasing the GPU quota**

Follow the instruction to make GPU quota increase, once you make the request google send you an email to make a 35$ payment or equivalent amount on the project you are making quota increase. Google is taking at least a day to accept the quota request once after the payment. Make GPU quota request](https://cloud.google.com/compute/docs/gpus/)

To test if your quota has been approved, you can run this

> gcloud beta compute regions describe us-east1

I requested 1 GPU of NVIDIA_K80_GPUS for which I paid 35$, that got credited into my cloud account.

limit: 1.0   
metric: *NVIDIA_K80_GPUS*  
usage: 0.0

**Step 3: Creating the GPU Instance**

> gcloud beta compute instances create gpu-deep-learner --machine-type n1-standard-2 --zone us-east1-d --accelerator type=nvidia-tesla-k80,count=1 --image-family ubuntu-1604-lts --image-project ubuntu-os-cloud --boot-disk-size 50GB --maintenance-policy TERMINATE --restart-on-failure

**Step 4: Installing CUDA - library to speedup NVIDIA GPU**

First  you need to ssh into the instance using gcloud

> gcloud compute ssh gpu-deep-learner --zone us-east1-d

Install CUDA 8.0 drivers:

> curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
>
> sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
>
> sudo apt-get update
>
> rm cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
>
> sudo apt-get install cuda -y
>
> nvidia-smi

Set environment variables:

> echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
>
> echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
>
> echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> ~/.bashrc
>
> source ~/.bashrc

**Step 5: Install cuDNN - Library runs on top of CUDA to speed up deep Neural network parallel computing.**

Register at cudnn[https://developer.nvidia.com/cudnn] and download cuDNN. Then, scp the file to your new instance.

Move the downloaded file to the home directory in my case it is /home/anveshtummala/Downloadedfile

*Note: We need to move the tgz file to the google instance, for this first we need to create the SSH key pairs using the following link [create sshKeyPair](https://cloud.google.com/compute/docs/instances/adding-removing-ssh-keys)*. Make sure you have the ssh keys already created by checking your Cloud instance using, first go to home with
>cd
>
>cd .ssh/
>
>ls   # this list out the ssh key pairs you already have


> gcloud compute scp gpu-deep-learner:~/.ssh/google_compute_engine* ~/.ssh/ --zone us-east1-d

Above line copies the ssh key pairs to my local home/account/.ssh/keyname. Later run the below command to move the tgz file from your local machine to google cloud
> scp -i .ssh/google_compute_engine cudnn-8.0-linux-x64-v7.1.tgz <external-IP-of-GPU-instance>:

Now, back to the remote instance and unzip the tgz file.

> cd
>
> tar xzvf cudnn-8.0-linux-x64-v7.1.tgz
>
> sudo cp cuda/lib64/* /usr/local/cuda/lib64/
>
> sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
>
> rm -rf ~/cuda
>
> rm cudnn-8.0-linux-x64-v7.1.tgz
>

This completes the NVIDIA/CUDA setup, install Python and other deep learning libraries. I prefer Anaconda to install packages.

**Step 6: Installing Anaconda and other libraries.**

> curl -O https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
>
>bash Anaconda3-4.3.1-Linux-x86_64.sh
>

If conda does not get recognized, set the environment variable
> export PATH="$HOME/anaconda3/bin:$PATH"

Create environment for gpu
> conda create -n gpu python=3.6
>
> source activate gpu

install tensorflow-gpu, keras

> pip install --ignore-installed --upgrade tensorflow-gpu

[Detailed instructions](http://inmachineswetrust.com/posts/deep-learning-setup/)

**Step 7: Configure Jupyter**

> conda install ipython
>
> conda install jupyter
>
> conda install scipy
>
> conda install seaborn
>
> conda install scikit-learn
>
> conda install keras
>

**Step 8: SSH tunnel forwarding**
> mkdir notebooks
> cd notebooks
> jupyter notebook

This gives the message that notebook is running at http://localhost:8888/?token=XXXXXXXXXXXXXXXX

Run this on the local machine to set up a tunnel from your local machine to access Jupyter over ssh.

> ssh -i .ssh/google_compute_engine -L 8899:localhost:8888 public_ip_of_instance

>Navigate to http://localhost:8899/


If you have problem in showing the new kernals, please do install > "conda install nb_conda_kernals"

Refrence: https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd


