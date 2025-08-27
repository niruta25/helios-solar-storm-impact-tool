# helios-solar-storm-impact-tool
A Solar Storm Impact Visualization Tool


Data Fields
Solar Flare (FLR) Data: Provides data on the location, start time, and intensity of solar flares. This is crucial for identifying the source of the storm on the sun's surface.

* flrID: Unique identifier for the flare
* beginTime: Start time of the flare
* peakTime: Peak time of the flare
* endTime: End time of the flare
* classType: X-ray class of the flare (e.g., C1.0, M5.0, X2.0)
* sourceLocation: Location on the sun where flare occurred
* activeRegionNum: Active region number
* instruments: List of instruments that detected the flare

Coronal Mass Ejection (CME) Data: This data tracks the massive eruptions of plasma from the sun, including their speed, trajectory, and estimated arrival time at Earth. This is the core data for predicting the storm's path.

* cmeID: Unique identifier for the CME
* startTime: Start time of the CME
* sourceLocation: Source location on the sun
* note: Additional notes about the CME
* instruments: List of instruments that observed the CME
* cmeAnalyses: Detailed analysis data including:
  * speed: Estimated speed of the CME
  * type: Type of CME
  * isMostAccurate: Whether this is the most accurate analysis



