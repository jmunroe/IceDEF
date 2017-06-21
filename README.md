# IceDEF

Iceberg Drift Ensemble Forecasting

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

#### EvanIceDEF
1. Anaconda or another Python environment
2. Data for sea-surface temperature, wind, and ocean velocities
```
file:///<your path>/IceDEF/WagnerModel/conditions/ECCO_20th/E2_sst_1992.mat
file:///<your path>/IceDEF/WagnerModel/conditions/ECCO_20th/E2_sst_1992.mat
```
These data files were too large to be stored on Github so they will have to be obtained by contacting one of the authors.

#### WagnerModel
1. MATLAB 2016a or higher
2. Data for sea-surface temperature, wind, and ocean velocities

```
file:///<your path>/IceDEF/WagnerModel/conditions/ECCO_20th/E2_sst_1992.mat
```
```
file:///<your path>/evankielley/IceDEF/WagnerModel/conditions/ECCO_20th/E2_sst_1992.mat
```
Again, these data files were too large to be stored on Github so they will have to be obtained by contacting one of the authors.


### Installation

The following instructions are tailored for Ubuntu 16.04.

#### EvanIceDEF

Enter the following to install Anaconda for Python 3.6:

```
bash ~/Downloads/Anaconda3-4.4.0-Linux-x86_64.sh
```

```
source ~/.bashrc
```

Change the root path in constants.py to:

```
root = /<your path>/WagnerModel
```

Run the program using Python 3:

```
python run.py
```

#### WagnerModel

With MATLAB 2016a or higher installed, change the root path in iceberg_shell.m to:

```
root = /<your path>/WagnerModel
```

Then simply execute iceberg_shell.m in the command window by typing:

```
iceberg_shell
```
<!---
### Viewing Output

#### EvanIceDEF



#### WagnerModel


## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

--->

<!---
## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 
--->

## Authors

* **James Munroe** - *Initial work* - [jmunroe](https://github.com/jmunroe)
* **Evan Kielley** - [evankielley](https://github.com/evankielley)

See also the list of [contributors](https://github.com/jmunroe/IceDEF/contributors) who participated in this project.

<!---
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

--->

## Acknowledgments

* Dr. Till Wagner for his model on Iceberg Drift
