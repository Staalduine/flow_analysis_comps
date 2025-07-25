# Flow Analysis

This repository is meant to take in videos of material moving through tube-like strucutures and be able to present the analysis results of multiple different methods to extract flow data. This problem can be hard, especially for the topic of interest for which this was written. The plan is for this repository to showcase common flow extraction methods, like PIV and kymograph analysis, along with particle tracking methods, and more particle-agnostic methods using Fourier filtering. 

## Topic of interest: Bidirectional nutrient flow.
Arbuscular Mycorrhizal Fungi (AMF) are fungi which appear to have a symbiotic relationship with plants, in which the plants will supply the fungus with excess carbon for growth, and the fungus in turn will provide nutrients from within the soil. This "trade" also has a spatial component to it, as the nutrients are generally away from the plant, in the soil, towards which the fungus must grow. Carbon then, has to flow from the plant to the fungla tips, whereas nutrients like phosphorus has to flow from those tips back to the plant. The implication is that there must be a network-wide bi-directional flow of material moving between plant and soil. This bi-directionality is also frequently observed within individual fungal hyphae. Videos taken with bright-field microscopes show material moving in seemingly opposite directions within single hyphal tubes, which makes the extraction of the flow dynamics a non-trivial problem. 

# Logging
The style for logging is as follows:
In the import statements add:
`from flow_analysis_comps.util.logging import setup_logger`

Then get the logger object using 
`self.logger = setup_logger(name="nameOfObject")`

Then use the logger object to do print statements