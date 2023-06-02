# PA bicycle crashes analysis, 2002-2021
 
## Introduction:
In this project we'll analyze data related to crashes involving bicycles in the state of Pennsylvania during the years 2002-2021. We focus on a publically accessible dataset of crash records in the state which is made available by Pennsylvania Department of Transportation (PENNDOT).

The central goal is to examine the prevalence of various aspects of vehicle crashes involving bicycles in Pennsylvania, and analyze how these factors might affect the severity of the crash from the cyclist's point of view.

An additional component of the project is to perform predictive modeling on the dataset using machine learning algorithms, with the goal of predicting whether a cyclist will suffer serious injury or fatality in a particular crash event based on features of that crash event.  This component is still in progress and will be added to the repository in a future update.

The PENNDOT dataset, as well as related resources such as a data dictionary, can be found (https://pennshare.maps.arcgis.com/apps/webappviewer/index.html?id=8fdbf046e36e41649bbfd9d7dd7c7e7e).

## Repository contents:

The project repository consists of the following components:
1. Two IPython files in the main directory:
    * 1_PA_bike_crashes_data.ipynb : A notebook which demonstrates the acquisition and cleaning of the dataset
    * 2_PA_bike_crashes_vis.ipynb : A notebook in which the data is analyzed and visualized in order to uncover patterns and inspire actions which might outcomes for cyclists
2. 'data' folder with the following subfolders:
    * 'raw_csv' : a directory containing four .CSV files which are processed in the first notebook
        * 'bicycles_raw.csv' : samples correspond to bicycle vehicles involved in crash events
        * 'crashes_raw.csv' : samples correspond to crash events
        * 'persons_raw.csv' : samples correspond to individuals riding bicycles involved in crash events
        * 'roadway_raw.csv' : samples correspond to roadways related to crash events
     * 'zip' : a directory intended to hold .ZIP files, if you choose to download them from the original PENNDOT page

## Summary of data analysis results:

1. The annual counts of crashes involving cyclists in PA showed a consistent downward trend since 2004, decreasing from above 1600 incidents to below 800 incidents in 2021.  However, the annual counts or crashes involving serious cyclist injury or fatality have not declined significantly.  In fact, in 2021 there were 103 crashes involving serious cyclist injury and 24 involving cyclist death - both the highest annual counts in this 20-year dataset!
2. Regarding the distributions of certain crash features and their relationship with cyclist injury severity:
    * The majority of cyclists in collisions are between 10-30 years of age.  However, older cyclists are overrepresented among cyclists suffering serious injury or fatality.
    * Around 75% of cyclists in collisions are traveling in a 25mph or below zone, presumably due to the prevalence of low speed limits in urban settings.  However, almost half of cyclists suffering serious injury or fatality were traveling in higher speed limit zones.
    * Midblock collisions were overrepresented among cyclists who suffered serious injury or fatality, possibly due to the higher vehicle speeds seen at midblock - 46% of cyclists suffering serious injury or fatality were in midblock collisions, as opposed to 35% of all cyclists.
    * 7.4% of cyclists involved in crashes suffered serious injury or fatality.  There are certain crash factors such that when we restrict to only crashes in which those factors are present, the percentage of cyclists with serious injury or fatality more than doubles (corresponding percentages in parentheses):
        * Involvement of at least one drugged driver (35.4%) or drinking driver (26.7%) 
        * Involvement of at least one heavy truck (22.4%) or commercial vehicle (17.5%)
        * The crash being speeding-related (21.3%)
        * The crash occuring in a dark unlit setting (20.9%) or at dawn (20%)
        * The crash occuring on a curved roadway (16.3%)
        * The crash occuring in a rural setting (15.9%)
    * When we restrict to crashes involving some pairs of two of these factors, the percentage of cyclists suffering serious injury or fatality surpassed 40%:
        * Speeding-related crashes with a drinking driver involved (51.43%)
        * Speeding-related crashes in dark unlit conditions (50%)
        * Speeding-related crashes on a curved roadway (44.4%)
        * Crashes involving a drinking driver on a curved roadway (42.9%)
        * Crashes on a curved roadway in dark unlit conditions (42.5%)

## Recommendations

Based on my findings, I would recommend the following actions to be taken in an effort to reduce the incidence of serious cyclist injury and cyclist fatality (as well as cyclist crashes in general) in Pennsylvania:

1. Bolstering cyclist education efforts regarding:
    * Safer riding practices around heavy trucks
    * Choosing routes with lower posted speed limits when possible
    * Visibility measures for low light riding conditions - reflectors, reflective clothing, headlights, taillights
    
2. Bolstering education efforts for private motorists and commercial vehicle drivers involving:
    * The serious risks involved with impaired driving and speeding
    * Awareness of cyclists and driving practices that help keep cyclists safe, especially when:
        * Driving in low light conditions
        * Driving in areas with higher posted speed limits
        * Driving during high-traffic times,e.g. morning and evening weekday commuting hours
        * Navigating curved roadways
3. Infrastructure improvements
    * Upgrading and/or repairing roadway lighting in areas where cyclists frequent, especially along curves and with attention to midblock areas.
    * Adding protected bicycle lanes/routes along roads commonly used by bicyclists, with a focus on:
        * routes with higher posted speed limits and/or where motor vehicle speeding is very prevalent
        * routes with significant use by heavy trucks and/or commercial vehicles
        * routes that are used heavily during the weekday morning and evening commutues
