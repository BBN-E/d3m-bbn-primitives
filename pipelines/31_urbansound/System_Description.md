1. System Name: BBN_SOU_Pipeline.jan2018.a.1
2. Point of Contact(s): Jan Silovsky (jan.silovsky@raytheon.com)
3. Institution: Raytheon BBN Technologies
4. System Description:

The system demonstrates application of self-organizing units (SOUs) for audio
signal classification.

At the time of submission of this system, the pipeline showcases an end-to-end
system for classification of audio signals, the primitives employed are not to be
considered as final. Their implementation is subject to future modifications
as is the choice of the employed primitives itself.

5. Primitives Covered:

d3m.primitives.bbn.time_series.ChannelAverager
d3m.primitives.bbn.time_series.SignalFramer
d3m.primitives.bbn.time_series.SignalDither
d3m.primitives.bbn.time_series.SignalMFCC
d3m.primitives.bbn.time_series.UniformSegmentation
d3m.primitives.bbn.time_series.SegmentCurveFitter
d3m.primitives.bbn.time_series.ClusterCurveFittingKMeans
d3m.primitives.bbn.time_series.SequenceToBagOfTokens
d3m.primitives.bbn.time_series.BBNTfidfTransformer

6. <TA2 Only> Primitive Execution Environment: N/A
7. <TA3 Only> TA3 Interface Technology: N/A
8. <TA3 Only> Remote Technology Requirements: N/A
9. <TA3 Only> User Training Materials: N/A
10. Outside Access: internet is required for the CI build. The system itself doesn't
require internet acess.

11. Learning Datasets:

Development experiments with the pipeline have been carried out with the 31_urbansound
seed dataset

12. References:

Will be provided in the future submission

