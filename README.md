# Textile Sorting with NIRSCAN

This repository contains code and resources for a textile sorting system using Near-Infrared (NIR) spectroscopy.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project leverages NIR spectroscopy to sort textiles on a conveyor belt. The system uses a NIR sensor to capture the spectral data of textiles and classifies them based on their material composition.

## Installation
To get started, clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/textile_sorting_NIRSCAN.git
cd textile_sorting_NIRSCAN
pip install -r requirements.txt
```

## Usage
1. **Data Collection**: Place textiles on the conveyor belt and start the NIR sensor to collect spectral data.
2. **Training**: Train the model using the collected data.
3. **Sorting**: Use the trained model to classify and sort textiles in real-time.

## Results
### Training Loss
![Training Loss](textile_sorting_NIRSCAN_\ML\images\loss.png)

### NIR Sensor Feedback
![NIR Sensor Feedback](textile_sorting_NIRSCAN_ML\images/output.png)

### Workflow Example
![Workflow Example](textile_sorting_NIRSCAN_ML\images/nirscan_example.jpg)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.