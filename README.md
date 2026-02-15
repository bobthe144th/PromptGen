# Prompt Dataset Generator Companion for TeichAI Datagen

Note: The model used in the script is relatively small and so will not generate extremely hard prompts as a bigger model might, for frontier distillation a bigger model will help massively.

This tool generates high-quality, domain-distributed training prompts designed specifically for use with TeichAI's Datagen CLI. The tool runs inside the Unsloth Docker container and uses Qwen 3 4B in GGUF format to create diverse prompts across ten customizable domains, producing output that feeds directly into TeichAI's reasoning trace generation workflow.

## TL;DR
Open Docker Desktop. Pull the unsloth:latest container. Run it with mounts. Go to the listed localhost port. Paste the code from script.py into a new notebook Cell and run it.

## Understanding the Purpose

TeichAI is doing remarkable work democratizing access to frontier AI reasoning capabilities. They're a small team of college students who distill powerful models like Claude Opus into smaller, locally-runnable versions that anyone can use. Their process involves three stages: creating diverse prompts, querying frontier models to capture reasoning traces, and fine-tuning open-source models on those traces. The challenge they face is that generating high-quality prompts at scale can be time-consuming, and since they're funding this research out of pocket, every dollar counts. Their Claude Opus dataset alone cost over fifty dollars to generate.

This tool addresses the first stage of their workflow by automating prompt generation completely. By using a local GGUF model through llama.cpp, you can generate thousands of diverse, high-quality prompts without any API costs. This means TeichAI and others working on similar projects can focus their limited budgets entirely on the expensive frontier model API calls that capture valuable reasoning traces, rather than spending money on prompt generation.

The tool is designed to integrate seamlessly into the existing TeichAI ecosystem. It runs in the same Unsloth container they already use for fine-tuning, outputs prompts in exactly the format their Datagen CLI expects, and requires no changes to their established workflow. You simply generate prompts with this tool, feed them into Datagen to get reasoning traces from frontier models, and then use those traces to fine-tune smaller models with Unsloth.

To learn more visit: teichai.com

## What You'll Be Setting Up

Before diving into installation, it helps to understand the technology stack you're working with. Docker Desktop provides a way to run containerized applications on Windows, giving you access to Linux environments without dual-booting or using a virtual machine. WSL2, which stands for Windows Subsystem for Linux version 2, is the underlying technology that makes this possible. It runs a real Linux kernel inside Windows, providing much better performance than the original WSL.

The NVIDIA Container Toolkit is what allows Docker containers to access your GPU. Without this, containers would only be able to use your CPU, which would make generation significantly slower. With the toolkit installed, the Unsloth container can leverage your NVIDIA GPU for fast prompt generation.

Unsloth itself is a framework designed specifically for efficient fine-tuning of language models. The Unsloth team has optimized various aspects of the training process to make fine-tuning faster and more memory-efficient than traditional approaches. Their Docker container comes pre-configured with all the necessary libraries and tools, including Jupyter Lab for interactive development. While you won't be doing fine-tuning for this prompt generation task, the Unsloth container provides a perfect environment because it already has the GPU drivers, Python libraries, and system dependencies you need.

Jupyter Lab is an interactive development environment that runs in your web browser. It allows you to write and execute Python code in notebooks, see results immediately, and document your work alongside your code. The Unsloth container includes Jupyter Lab, which means you'll have a familiar, user-friendly interface for running the prompt generation script and exploring the results.

## Prerequisites

Before you begin the installation process, you need to verify that your system meets the minimum requirements. You'll need a Windows 10 or Windows 11 system with virtualization support enabled in your BIOS. Most modern computers have this enabled by default, but if you encounter issues during WSL2 installation, you may need to check your BIOS settings. You should also have at least sixteen gigabytes of RAM, though the tool can work with less if you're patient. Eight gigabytes is the absolute minimum.

If you want to use GPU acceleration, which significantly speeds up prompt generation, you'll need an NVIDIA GPU with CUDA support. The GGUF format is quite efficient, so even a GPU with six gigabytes of VRAM will work well. If you don't have an NVIDIA GPU, the tool will still run on CPU, just more slowly. You'll generate about fifteen to thirty prompts per minute on a modern multi-core CPU versus sixty to one hundred prompts per minute with a decent GPU.

You should also have at least twenty gigabytes of free disk space. The Docker Desktop installation, WSL2 Linux distribution, Unsloth container image, and model files all require storage. The GGUF model itself is about two and a half gigabytes, but you want buffer space for Docker's image layers and other files.

## Step One: Installing Docker Desktop

Docker Desktop is your gateway to running the Unsloth container on Windows. It provides a user-friendly interface for managing containers and integrates cleanly with WSL2 to give you excellent performance. The installation process is straightforward, but there are a few important details to understand along the way.

Start by visiting the official Docker website at docker.com and navigating to the Docker Desktop download page. You'll find a prominent download button for Windows. The installer is around five hundred megabytes, so the download might take a few minutes depending on your internet connection. Once the download completes, run the installer. During the installation process, you'll see an option to use WSL2 instead of Hyper-V. Make absolutely certain this option is checked. WSL2 provides much better performance and integration than Hyper-V for our purposes, and it's required for the NVIDIA Container Toolkit to work properly.

The installation will take several minutes as it extracts files and configures your system. When it finishes, you'll need to restart your computer. This restart is necessary because Docker Desktop needs to configure Windows services and WSL2 integration that can only be activated with a reboot. After your computer restarts, Docker Desktop should launch automatically. If it doesn't, you can find it in your Start menu and launch it manually.

When Docker Desktop starts for the first time, you'll see a welcome screen that might ask you to sign in or create a Docker account. You can skip this step for now since you don't need a Docker account to use Docker Desktop locally. What matters is that Docker Desktop is running, which you can verify by looking for the Docker icon in your system tray. The icon looks like a whale carrying shipping containers. When you hover over it, it should say "Docker Desktop is running."

## Step Two: Installing WSL2 and Configuring It

WSL2 is the foundation that makes everything else work smoothly. Windows Subsystem for Linux version 2 represents a significant leap forward from the original WSL because it runs a real Linux kernel, providing full system call compatibility and much better performance. Installing and configuring WSL2 properly is essential for getting the best experience with the Unsloth container.

Open PowerShell as an administrator. You can do this by searching for PowerShell in the Start menu, right-clicking on it, and selecting "Run as administrator." In the PowerShell window, run the command `wsl --install`. This single command does a lot of work behind the scenes. It enables the WSL feature in Windows, downloads and installs the Linux kernel update, and installs Ubuntu as your default Linux distribution. The process takes several minutes and requires an internet connection since it's downloading the Linux distribution.

When the installation completes, you might be prompted to restart your computer again. After restarting, Ubuntu will automatically launch and ask you to create a username and password. Choose something you'll remember, as you'll use these credentials whenever you need to run commands with elevated privileges inside the Linux environment. The username doesn't need to match your Windows username, and you can make it something simple like "user" if you prefer.

Once you've set up your Ubuntu credentials, you have a working WSL2 environment. You can verify this by opening PowerShell again and running `wsl --list --verbose`. This command shows you all installed Linux distributions and their WSL version. You should see Ubuntu listed with version 2. If for some reason it shows version 1, you can upgrade it by running `wsl --set-version Ubuntu 2`, though this shouldn't be necessary with a fresh installation.

## Step Three: Installing the NVIDIA Container Toolkit

If you have an NVIDIA GPU and want to use it for accelerated prompt generation, you need the NVIDIA Container Toolkit. This toolkit enables Docker containers to access your GPU hardware, which is essential for running CUDA-accelerated applications inside containers. The installation process requires working inside your WSL2 Ubuntu environment.

Microsoft provides excellent documentation for this process, and rather than duplicate it here, I'll point you to their official guide. Open your web browser and navigate to the Microsoft documentation for enabling NVIDIA CUDA on WSL2. You can find this by searching for "NVIDIA CUDA on WSL2 Microsoft docs" or by going directly to docs.microsoft.com and searching for the CUDA on WSL guide. The documentation walks through installing the NVIDIA drivers on your Windows host, which is the first critical step. You need to have recent NVIDIA drivers installed on Windows itself before the container toolkit will work inside WSL2.

After ensuring your Windows NVIDIA drivers are up to date, you'll follow the instructions to install the NVIDIA Container Toolkit inside your WSL2 Ubuntu distribution. This involves adding the NVIDIA package repository to Ubuntu's package manager, updating the package list, and installing the toolkit package. The commands look something like this, though you should follow the official documentation for the exact current commands:

First, you set up the repository by downloading the GPG key and adding the package repository to your system. Then you update your package list to include the new repository. Finally, you install the nvidia-container-toolkit package and restart the Docker daemon. The Microsoft documentation provides the exact commands for each step and explains what each one does.

To verify that everything is working correctly, you can run a test command that launches a small CUDA container. The command is `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`. If this runs successfully and shows your GPU information, you know the toolkit is installed correctly and containers can access your GPU. If you see any errors at this stage, the Microsoft documentation includes troubleshooting steps for common issues.

If you don't have an NVIDIA GPU or choose not to set up GPU acceleration, you can skip this entire step. The prompt generation tool will work fine on CPU, just with slower generation speeds. You'll simply omit the `--gpus all` flag when running containers, and llama.cpp will automatically use your CPU instead.

## Step Four: Getting the Unsloth Container

Now that you have Docker Desktop, WSL2, and optionally the NVIDIA Container Toolkit all configured, you're ready to get the Unsloth container. Unsloth is a framework specifically designed for efficient fine-tuning of language models. The team behind Unsloth has optimized the training process to be significantly faster and more memory-efficient than standard approaches using vanilla PyTorch and Transformers. While this tool doesn't actually perform fine-tuning, the Unsloth container provides an ideal environment because it comes pre-configured with all the necessary dependencies, GPU drivers, and Python libraries you need for working with language models.

The Unsloth team maintains an official Docker image on Docker Hub, which is a public registry for container images. To download this image, open a terminal window. You can use PowerShell, Windows Terminal, or even the Ubuntu terminal from WSL2. The command is the same regardless: `docker pull unsloth/unsloth:latest`. This tells Docker to download the latest version of the Unsloth image from Docker Hub.

The download will take some time because the Unsloth image is several gigabytes. You'll see progress bars showing each layer being downloaded. Docker images are built in layers, which is why you see multiple items downloading. This layered approach is actually quite clever because it means when you update the image later, Docker only needs to download the layers that changed rather than the entire image again.

Once the download completes, you can verify the image is ready by running `docker images`. This command lists all the Docker images stored on your system. You should see `unsloth/unsloth` in the list with a size of several gigabytes. The image includes a complete Ubuntu Linux environment with Python, PyTorch, CUDA libraries, and various other tools pre-installed and configured. This is what makes Docker so powerful: the Unsloth team has done all the difficult setup work, and you get a ready-to-use environment by simply downloading their image.

## Step Five: Understanding Unsloth and Jupyter Lab

Before we launch the container and start using it, it's worth taking a moment to understand what Unsloth provides and how Jupyter Lab fits into the picture. This context will help you work more effectively with the environment.

Unsloth is fundamentally about making fine-tuning accessible. Training or fine-tuning large language models typically requires expensive hardware and significant technical expertise. The Unsloth team has built optimizations that make this process faster and more memory-efficient. They've implemented custom CUDA kernels, optimized memory layouts, and clever batching strategies that can speed up training by two to five times compared to standard implementations. They also focus on making the fine-tuning process more approachable by providing clear examples and good documentation.

For this prompt generation project, you're not actually doing fine-tuning, but the Unsloth container is still ideal because it provides a stable, well-configured environment for working with language models. It has the right versions of PyTorch, CUDA, and other libraries all properly configured to work together. You don't have to worry about dependency conflicts or driver issues because the Unsloth team has already solved those problems in their container image.

The Unsloth container includes Jupyter Lab, which is an interactive development environment that runs in your web browser. If you've used Jupyter Notebooks before, Jupyter Lab is the next evolution of that concept. It provides a more flexible interface where you can have multiple notebooks, text files, and terminals open simultaneously in tabs and panels. You write code in cells, execute those cells individually, and see the results immediately below the code. This interactive workflow is perfect for data science and machine learning tasks because you can experiment, visualize results, and iterate quickly.

When you run the prompt generation script, you can do it either from the command line inside the container or from a Jupyter notebook. The command line approach is more straightforward for running the script start to finish, but for the casual user Jupyter notebooks are generally better. Both approaches work equally well for generating your final prompt datasets.

## Step Six: Launching the Unsloth Container

Now you're ready to launch the Unsloth container and start working with it. The way you launch the container determines how it behaves, what resources it can access, and how you interact with it. Understanding the launch command and its various flags will help you use the container effectively.

The basic command to launch the container looks like this, though we'll break down each part to explain what it does:

```bash
docker run -it --gpus all -v $(pwd):/workspace -p 8888:8888 --name teichai-prompts unsloth/unsloth:latest
```

Let's examine each component of this command to understand what it's doing. The `docker run` part tells Docker you want to create and start a new container from an image. The `-it` flags are actually two separate flags combined: `-i` keeps the container's standard input open so you can type commands, and `-t` allocates a pseudo-terminal, giving you a proper interactive shell. Together, these flags give you an interactive terminal session inside the container.

The `--gpus all` flag tells Docker to give the container access to all your GPUs. This is where the NVIDIA Container Toolkit comes into play. If you skipped installing the toolkit or don't have an NVIDIA GPU, you would omit this flag entirely. The container will still run fine, just without GPU acceleration.

The `-v $(pwd):/workspace` flag creates a volume mount, which is how you share files between your Windows system and the container. The `$(pwd)` part expands to your current directory on the Windows side, and `/workspace` is where that directory appears inside the container. This means any files you create in `/workspace` inside the container are actually saved to your Windows directory and persist after the container stops. This is crucial because without this mount, any files you create inside the container would be lost when the container stops.

The `-p 8888:8888` flag maps port 8888 from inside the container to port 8888 on your host system. This is necessary for Jupyter Lab to work. Jupyter Lab runs a web server inside the container on port 8888, and this port mapping allows you to access that web server from your Windows browser by navigating to `localhost:8888`.

The `--name teichai-prompts` flag gives your container a friendly name. Without this, Docker would assign a random name like "quirky_einstein" or "suspicious_tesla." With the name specified, you can easily reference this container in future commands like `docker start teichai-prompts` or `docker stop teichai-prompts`.

Finally, `unsloth/unsloth:latest` tells Docker which image to use for creating this container. The `:latest` tag means you want the most recent version of the Unsloth image.

When you run this command, Docker creates a new container from the image and drops you into a bash shell inside the container. Your prompt will change to something like `root@containerid:/workspace#`, indicating you're now working inside the container's Linux environment. From this point forward, any commands you type are executed inside the container, not on your Windows system.

## Step Seven: Setting Up the Environment Inside the Container

Once you're inside the container, you need to install a few additional dependencies that aren't included in the base Unsloth image. The Unsloth container has most of what you need, but it doesn't include llama-cpp-python or huggingface-hub, which are specifically required for working with GGUF models.

The installation of llama-cpp-python is slightly complex because it needs to be compiled with CUDA support to enable GPU acceleration. When you install a Python package with pip, it usually just downloads pre-compiled binary files. But llama-cpp-python needs to be compiled specifically for your system to take advantage of your GPU. This is done by setting an environment variable that tells the build system to include CUDA support.

If you have GPU support enabled, run this command:

```bash
pip install llama-cpp-python
```
If you do not have a GPU the code is the same.

The huggingface-hub library will automatically cache downloaded model files, so you only download the Qwen 3 4B GGUF model once, even if you stop and restart the container later.

## Step Eight: Running the Prompt Generation Script

Now that your environment is fully configured, you need to get the actual prompt generation script into the container. To do this paste the content of "script.py" into your Notebook or Command line.

Press Shift + Enter in the Notebook or Enter in the Command line to run the pasted script.

When you first run the script, several things happen in sequence. First, the script will download the Qwen 3 4B GGUF model file from HuggingFace if it's not already cached. The specific file it downloads is `Qwen3-4B-Q4_K_M.gguf`, which is about two and a half gigabytes. You'll see a progress bar showing the download status. This download only happens once because huggingface-hub caches the file in your container's home directory.

After the model downloads, llama.cpp loads it into memory. You'll see a message confirming the model loaded successfully, along with information about how many GPU layers are being used. If you see "GPU layers: all," that means the model is fully loaded on your GPU and generation will be fast. If you see "GPU layers: 0," it means the model is using CPU only.

The script then prompts you for configuration. First, it asks how many total prompts you want to generate. For your first test run, start with something modest like fifty or one hundred prompts. This lets you verify everything works and evaluate the quality of the generated prompts without committing to a long generation session.

Next, the script asks you to specify the percentage distribution across ten domains: Coding, Math, Science, Web Development, Data Science, Machine Learning, Creative Writing, Logic, Reasoning, and General Knowledge. You need to enter a percentage for each domain, and they must sum to exactly one hundred. The script validates this and will tell you if the sum is off.

Here's an example balanced distribution you might use:

For Coding, you might allocate twenty percent if you want a good representation of programming tasks. For Math, fifteen percent gives you a solid foundation of mathematical reasoning. Science could be ten percent to cover various scientific domains. Web Development might get fifteen percent since it's a distinct skillset from general coding. Data Science and Machine Learning could each get ten percent, as these are specialized but important areas. Creative Writing might only need five percent since it's less central to technical reasoning datasets. Logic and Reasoning could each get five percent, though you might increase these if you're focused on reasoning capabilities. General Knowledge rounds it out at five percent to ensure the model has basic factual knowledge.

The key is thinking about what capabilities you want in the models that will eventually be trained on these prompts. If you're creating a dataset for mathematical reasoning, you might give Math forty percent and Logic and Reasoning another fifteen percent each. If you're creating a dataset for coding assistance, you might give Coding and Web Development thirty percent each.

After you enter the domain percentages, the script asks for an output filename. The default is `prompt_dataset.md`, but you can name it anything you want. If you're generating multiple datasets with different configurations, use descriptive names like `math_heavy_500.md` or `balanced_coding_1000.md` to keep them organized.

Once you've provided all the configuration, the script begins generating. You'll see progress messages as it works through each domain. For each batch of prompts, it shows you "Generating batch of X prompts for [domain]... ✓ (Y prompts)" where Y is how many valid prompts were extracted from the model's output. The script generates prompts in batches of ten by default, which balances efficiency with memory usage.

After all domains are processed, the script checks for duplicates. It uses case-insensitive comparison, so "Write a function to reverse a string" and "write a function to reverse a string" would be considered duplicates. If it finds more than ten duplicates, it automatically regenerates prompts to replace them, maintaining your target count.

Finally, the script saves all unique prompts to your specified output file. This file appears in your workspace directory, which is mounted from your Windows system, so you can access it directly from Windows File Explorer or any text editor.

## Understanding the Output

The output from the prompt generation script is deliberately simple: a plain text markdown file with one prompt per line. This simplicity is intentional because it makes the file easy to process with any tool, easy to inspect manually, and perfectly compatible with TeichAI's Datagen CLI.

When you open the generated file, you'll see prompts that look like this:

```
Write a Python function to implement binary search on a sorted array
Explain the difference between supervised and unsupervised learning with examples
Design an experiment to test the effect of temperature on enzyme activity
Create a REST API endpoint for user authentication using JWT tokens
How do you handle missing data in a pandas DataFrame
Implement a neural network for image classification using PyTorch
Write a short story about a detective solving a mystery in Victorian London
Determine the validity of this logical argument using truth tables
A train leaves Station A at 60 km/h while another leaves Station B at 40 km/h, when do they meet
What are the main causes of World War I
```

Each prompt is self-contained and ready to be sent to a frontier model for reasoning trace generation. The prompts vary in complexity from simple factual questions to more involved tasks requiring multi-step reasoning. This variety is important because it creates a diverse training dataset that covers different types of thinking.

You can verify the domain distribution by counting prompts or by looking at the patterns in the content. Coding prompts typically mention programming languages, functions, or algorithms. Math prompts involve calculations or mathematical concepts. Science prompts reference scientific phenomena or experiments. The distribution should roughly match the percentages you specified, though there may be small variations due to rounding.

## Next Steps: 

##Using with TeichAI's Datagen

Once you have your prompt dataset, you're ready to move to the next stage of the TeichAI workflow. The prompts you generated are the first ingredient in creating a reasoning trace dataset. To complete the process, you'll use TeichAI's Datagen CLI to send these prompts to a frontier model like Claude Opus, GPT-4, or Gemini.

TeichAI's Datagen tool takes your prompt file as input and generates a JSONL file where each line contains a JSON object with the prompt, the frontier model's response, and metadata. The frontier model's responses include detailed reasoning traces showing step-by-step how the model thinks through the problem. These reasoning traces are the valuable part because they teach smaller models how to approach problems systematically rather than just memorizing answers.

After you have the JSONL file with reasoning traces, you use Unsloth to fine-tune a smaller open-source model on this data. The fine-tuning process teaches the smaller model to replicate the reasoning patterns of the frontier model. The result is a distilled model that captures much of the frontier model's reasoning capability but runs locally on consumer hardware.

This complete workflow transforms your free, locally-generated prompts into a valuable training dataset and ultimately into an accessible, locally-runnable model. By automating the prompt generation stage, you can create larger, more diverse datasets while keeping costs focused on the frontier model API calls where the real value is created.

## Troubleshooting Common Issues

Even with careful setup, you might encounter some issues. Understanding the most common problems and their solutions will help you work through any difficulties.

If Docker Desktop won't start or shows an error about WSL2, verify that WSL2 is properly installed by running `wsl --list --verbose` in PowerShell. You should see at least one distribution listed with version 2. If WSL2 isn't installed or is using version 1, go back and reinstall it following the official Microsoft documentation.

If the container can't access your GPU despite having the NVIDIA Container Toolkit installed, first verify your Windows NVIDIA drivers are up to date. Then check that the toolkit is properly installed in WSL2 by running the test command `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` from inside your WSL2 Ubuntu terminal. If this test fails, you'll need to reinstall the NVIDIA Container Toolkit following the Microsoft documentation carefully.

If the model download is very slow or fails, this is usually a network issue with HuggingFace. The GGUF file is about two and a half gigabytes, so on a slower connection it can take twenty to thirty minutes. If the download keeps failing, try again at a different time when network conditions might be better. The file is cached once downloaded, so you won't need to download it again.

If generated prompts seem low quality or too similar to each other, you can adjust the temperature parameter in the script. Look for the `temperature=0.9` setting in the `generate_prompts_batch` method and try increasing it to 1.0 or even 1.1 for more creative variation. You can also try using a different quantization of the GGUF model, like Q6_K instead of Q4_K_M, which preserves more of the original model's capabilities at the cost of slightly higher memory usage.

If Jupyter Lab won't start or you can't access it from your browser, verify that you mapped the port correctly when launching the container with `-p 8888:8888`. Also make sure you're using the complete URL with the token that Jupyter Lab prints when it starts. Without the correct token, you won't be able to access the interface.

## Tips for Success

Start small with your first generation run to verify everything works correctly. Generate fifty to one hundred prompts with a balanced domain distribution, then review the output manually. Look for quality, diversity, and whether the prompts match what you were expecting. This initial test helps you catch any issues before committing to generating thousands of prompts.

Pay attention to the domain distribution you choose. The distribution should reflect the capabilities you want in the models that will eventually be trained on these prompts. If you're creating a dataset for general reasoning, use a balanced distribution. If you're targeting a specific capability like mathematical reasoning or coding assistance, weight those domains more heavily.

Consider generating multiple smaller datasets with different configurations rather than one massive dataset. For example, you might generate one dataset that's coding-heavy, another that's math-heavy, and a third that's balanced. This gives you flexibility when it comes time to fine-tune models, as you can combine datasets in different proportions or use them separately for specialized models.

Monitor your GPU usage during generation with `nvidia-smi` if you're using GPU acceleration. This helps you verify the GPU is actually being used and shows you how much memory the model is consuming. If you're consistently hitting memory limits, you might need to reduce batch size or use a smaller quantization of the model.

Keep your generated prompt files organized with descriptive names that indicate the configuration used. Include the total number of prompts and the date in the filename, like `balanced_1000_2024_12.md`. This makes it much easier to keep track of multiple datasets and understand what each one contains without opening them.

## Contributing to TeichAI's Mission

This tool was created specifically to support TeichAI's mission of democratizing access to frontier AI reasoning capabilities. If you use this tool to generate prompts and create datasets, consider sharing your experience with the broader community. The open-source AI community benefits when people share what works, what doesn't, and how tools can be improved.

If you create a particularly good dataset using this workflow, consider publishing it on HuggingFace so others can benefit. The more high-quality open datasets available, the more people can train capable models without needing massive resources. Every contribution to the ecosystem helps make AI more accessible.

The ultimate goal is to make frontier reasoning capabilities accessible to everyone, not just large labs with massive computing budgets. By contributing tools, datasets, and knowledge to the open-source community, you're helping achieve that goal. Whether you're using this for personal projects or as part of research, you're participating in a broader movement toward democratized AI.
