# jba-flood-events

Process JBA flood events for exposure analysis

## Setup

Requirements are Python and packages listed in `requirements.txt`.

To set up a virtual environment using venv:

```
python3 -m venv ./venv
. venv/bin/activate
pip install -r requirements.txt
```

## Data

Sample data for Jamaica to be shared directly.

Test data extract might be useful.

Global data is on the SoGE server at `/soge-home/projects/mistral/JBA_flood_datasets/global_2021`


## Method notes

The input is mainly a set of observation points (OP) and for each OP
we have 10,000-years of return period data. More specifically:

- observation points, separately for fluvial and pluvial flooding, which are the
  locations of event observations
- events, which have an event ID linked to one or more observation points, with
  a return period value for each of those points
- return period maps for a set of fixed return periods, which give flood depths
  across the whole region
- hydrological accumulation zones (HAZ) within which flooding is likely to be
  homogenous, used to link observation points to their surrounding area

The first step is to associate the flood depths at the locations of each OP. For
that we refer to the return period flood maps. From JBA’s technical note these
are the steps we need to follow to create the (return period, flood depth) value
are each OP for all event sets:

- The definition of hazard events include – all related OPs with return periods
  greater than two years, and at least one OP with return period of at least 10
  years. I think this sounds reasonable.

- There is something called a Hazard Onset Return Period (HORP), which is the
  minimum return period for assuming the onset of inundation. The default JBA
  view is that a river begins flowing out of bank at return periods in excess of
  two years. The lowest mapped return period in JBA's Global Flood Map is 20
  years. Exposure points inundated by the 20-year flood (and by definition not
  the 2-year, in-bank flow) are therefore assumed to begin inundating in the
  11-year flood. This is the default value of the HORP parameter. This parameter
  can be changed by the user to reflect their knowledge of their exposure, or to
  test sensitivity of losses to the assumption. So basically, they assume that
  flood depth = 0 at return periods ≤ 11-years. Do we want to keep the same for
  our study? To be honest I am a bit confused why it is 11-years and not
  2-years, given that they assume the river starts overtopping at 2-years. Am I
  missing something?

- The assigned flood depth ( dy) at an OP for any given return period ( ry), is
  estimated by intersecting at OP with return period flood maps and then using
  Equation (1) to interpolate between any two flood depths (dl,du) between
  return periods (rl,ru) in the range 𝒓= 11,20,50,100,200,500,1500 .

```latex
$d_{y} = \left\{
  \begin{array}{ c l }
    0 & \quad \textrm{if } r_{y} \leq 2 \\
    d_{l} + \left( d_{u} - d_{l} \right) \frac{\log{r_{y}} - \log{r_{l}}}{\log{r_{u}} - \log{r_{l}}} & \quad \textrm{if } r_{l} < r_{y} \leq r_{u}, \forall r_{l}, r_{u} \in \mathbf{r} \\
    d_{max} & \quad \textrm{if } r_{y} > r_{max}
  \end{array}
\right.$
```

![Equation 1](images/eq1.svg)

<!--
Note on equation image, using github math rendering service:
- this might stop working! could fall back to local latex
- start with "https://render.githubusercontent.com/render/math?math="
- then include latex, all on one line (delete newlines)
- delete start/end "$" symbols
- replace space " " with "%20"
- replace       "\" with "%5C"
- replace       "&" with "%26"
-->

The above assumes that the flood depth > 0 when the OP’s are intersected with
the return period maps.

If that is not the case, then the first return period at which flood depth > 0
is taken along with the return period map value below it, and it is assumed that
the lowest return period for flood depth = 0 is the mean of the two values.
Example: if at an OP the first return period with flood depth greater than 0
corresponds to 50-years, then flood depth = 0 is assumed for a return period of
35 – the mean of 20 and 50. I think this assumption is fine.

Once we have the (return period, flood depth) values for each OP for each event
we will then need to create spatial footprints for the flood events to
extrapolate towards points, lines and areas of interest for us. For me this is
the tricky bit.

The process outlined in the JBA technical description is as following:

The OPs can be linked to Hydrological Accumulation Zones (HAZ), which provide
geographic areas for the purpose of transferring observation point event
severities to exposure points (EP). Exposure points are attributed with the IDs
of the HAZ polygon that contains them and all OPs within the HAZ. HAZ polygons
are boundaries of hydrological sub-catchments that are capped in size. It is
assumed that the return period for an event is approximately homogenous
throughout a given HAZ; therefore, they are a convenient way to transfer event
severities from river and precipitation
gauges (OPs) to exposure points.

There will be several possibilities of OP, EP, HAZ intersections and
combinations.

- If 1 HAZ has only one OP, then all EPs are assigned the return period of
  the OP.

- If 1 HAZ has multiple OPs, then the EPs are assigned the geometric mean of
  all OP return period values.

- When there is no OP in a HAZ that contains EPs, event severity is taken
  from the OP closest to the HAZ border within a configurable distance. If no
  OPs are within this distance for a given flood type, the exposure point is
  considered not at risk from that flood type.

- The value of 100km is chosen for river flood based on the assumption that
  catchments might be similar within 100km.

- 99% of HAZs are joined to a surface water OP within the default distance.
  The default search distance for precipitation intensity is 1,000km. In
  practice most OPs are not joined over such a long distance - 1,000km is
  used to capture offshore islands, long peninsulas or similar. The 1,000km
  limit also ensures that rainfall events are not transferred unreasonably
  across oceans to remote islands, where there is explicitly no GFES
  coverage.

- Where an EP is not located within a HAZ geographic area for RP transfer, it
  is not analysed.

From JBA technical notes EP is a point but for us the method has to be
extended to points, lines and areas of assets. I think the best way to do
this would be to do the following.

- Select the EP as the centroids of the flood map raster grid cells with at
  least 1 value of flood  depth > 0 extracted across all the return period
  maps.

- For each EP generate the (return period, flood depth) tuple-list across all
  events by applying steps 1-6 above.

- For a point asset within the raster grid, the same (return period, flood
  depth) tuple-list is then applied.

- For a line asset the tuple-list (return period, flood depth, length
  intersected) is created, where the length intersected has to be estimated
  only once as it is the length of the line segment intersecting with the
  grid geometry (square).

- For a polygon asset the tuple-list (return period, flood depth, area
  intersected) is created, where the area intersected has to be estimated
  only once as it is the area of the polygon intersecting with the grid
  geometry (square).