# IANNwTF
Project for IANNwTF Course WiSe 21/22 Cognitive Science at University Osnabrück 
Members: Peter Keffer, Linda Ariel Ventura di Lorenzo Lopes, Erik Bossow, Julia Fülling, Jan-Eric Wiemann, Lena Kagoshima

<!-- ABOUT THE PROJECT -->
## About The Project

[![Neural Style Transfer Example][Doggo_Art.png]]

There are many great README templates available on GitHub; however, I didn't find one that really suited my needs so I created this enhanced one. I want to create a README template so amazing that it'll be the last one you ever need -- I think this is it.

Here's why:
* Your time should be focused on creating something amazing. A project that solves a problem and helps others
* You shouldn't be doing the same tasks over and over like creating a README from scratch
* You should implement DRY principles to the rest of your life :smile:

Of course, no one template will serve all projects since your needs may be different. So I'll be adding more in the near future. You may also suggest changes by forking this repo and creating a pull request or opening an issue. Thanks to all the people have contributed to expanding this template!

Use the `BLANK_README.md` to get started.


This project is part of the _'Introduction to Artificial Neural Networks with Tensorflow'_ course at the _University Osnabrück_.
In our project we generate pictures to which the art style of a chosen artist is transmitted while keeping the pictures content complete.
For this we implemented 2 different approaches, on the one hand CycleGAN ( Link ) and on the other hand Neural Style Transfer ( Link ), with the use of TensorFlow.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- Project Structure -->
## Project Structure

    .
    ├── src                    
    │   ├──── Configs          
    │   │     ├── Config.py
    │   │     ├── Config_CycleGAN.py
    │   │     ├── Config_NeuralStyleTransfer.py
    │   │
    │   ├──── Data   
    │   │     ├── Downloads
    │   │     ├── DataPipeline.py
    │   │     └── Dataset_Downloader.py
    │   │
    │   ├── Logs
    │   │
    │   ├──── Models
    │   │     ├── BaseModel.py
    │   │     ├── CycleGAN.py
    │   │     ├── NeuralStyleTransfer.py
    │   │     └── PatchGAN.py
    │   │
    │   ├──── Utilities
    │   │     ├── BaseModel.py
    │   │     ├── CycleGAN.py
    │   │     ├── NeuralStyleTransfer.py
    │   │     ├── PatchGAN.py
    │   │
    │   └── wandb                
    │── .gitignore
    │── LICENSE
    │── main.ipynb
    │── README.md
    │── requirements.txt

<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.

  ```
  pip install -r requirements.txt
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- AUTHORS -->
## Authors
* Erik Bossow
* Julia Fülling
* Lena Kagoshima
* Peter Keffer
* Linda Ariel Ventura di Lorenzo Lopes
* Jan-Eric Wiemann


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact



<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!



<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
