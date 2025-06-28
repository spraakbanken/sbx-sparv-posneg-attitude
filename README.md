#  sparv-sbx-posneg-attitude

A plugin for [Sparv](https://spraakbanken.gu.se/sparv/) that annotates positive and negative attitude in Swedish text, more specifically the object that the attitude is directed toward. 

Example: Jag_O älskar_O glass_POS med_POS strössel_POS

(The author of this text expresses a positive attitude toward ice-cream.)

This plugin uses a model based on [KB-BERT](https://huggingface.co/KB/bert-base-swedish-cased), which can 
be found [here](https://huggingface.co/sbx/KB-bert-base-swedish-cased_posneg_attitude). The model was trained 
on shorter texts, so performance may vary in other domains. Note that the quality of the annotations may vary in general.


## Install

1. Install [Sparv](https://spraakbanken.gu.se/sparv/):  
```
pipx install sparv
sparv setup
```

2. Clone this plugin's repository:  
```
git clone https://github.com/spraakbanken/sparv-sbx-posneg-attitude
```

3. Inject the plugin into Sparv:  
```
pipx inject sparv /YOUR/PATH/TO/THE/PLUGIN/sparv-sbx-posneg-attitude
```

_In case you need to uninstall the plugin:_  
```
pipx runpip sparv uninstall sparv-sbx-posneg-attitude
```

## Usage

In your `config.yaml` file for corpus configuration, specify `sbx_posneg_attitude.attitude`, in order to apply the plugin on your text.

```
export:
  annotations:
    - <token>
    - <token>:sbx_posneg_attitude.attitude
```

