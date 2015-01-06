# Hierarchical phrase-based translation decoder

This is a hierarchical phrase-based translation decoder which supports parallel decoding. New feature functions can also be implemented easily.

## Prerequisites

This decoder has been tested on Ubuntu 14.04 and Mac OS 10.9. But it should work on other platforms which supported by the following softwares:

* Python 2.7.8 or later (Python 3.x is not supported)

## Files

* alg.py: common utility algorithms
* errors.py: exception classes
* ioutil.py: IO utilities
* lm.py: srilm wrapper 
* lru.py: LRU cache
* timeutil.py: timer
* decoding: decoding related code
	* translator.py: decoder entry
	* chart.py: chart
	* ckydecoder.py: decoder
	* config.py: class for parsing decoder configure file
	* recombination.py: recombination checkers
	* rulefilter.py: a tool for filtering rule table given input file
	* rules.py: translation rule table
	* feat: non-standard features
		* feature.py: interface
		
## Decode

The decoder entry is `decoding/translator.py`, the parameters are as following:

* -c, --config: decoder configure file (required), see decoder.config.template for example
* -k, --kbest: size of kbest list, default value is 1
* --drop-oov: drop OOV words in the translations
* -d, --debug: output debug information
* -f, --features: output feature values
* -i, --input: input file, one pure-text sentence per line
* -o, --output: output file, the format is `sentence_id ||| translation ||| feature values ||| score`
* --checking: check feature values, for debugging
* --expend-loser: do not touch it unless you know what you are doing
* --with-rule-tree: output rules used in producing a translation
* -t, --threads: how many processes to use in decoding, default value is 1
* -l, --logger-config: logger config file, default value is None. Please see logger.json.template for example

Note: if `--threads` is set to a value bigger than 1, the translations will not be written until all the sentences have been decoded. Please be patient.

## Implement New Feature Functions

If you want to implement a new feature function, you only need to touch two pieces of code

1. Implement your feature function as a subclass of `decoding.feat.Feature`. You need to implement the following two methods:
	* `get_log_value()`: compute feature value
	* `should_recombined()`: check whether two hypotheses should be recombined
2. Register the new feature function in `decoding.translator.build_extra_feature_funcs`.

## Demo

You can find demo data at <http://nlp.csai.tsinghua.edu.cn/~lpeng/software/hiero-decoder-py/hiero-decoder-py-demo.zip>.


