import { ResponsiveLine } from '@nivo/line'
import { InlineMath } from 'react-katex';

const LineGraph = ({totalLen, twissData, setZValue, beamline, twissAxis, scroll, setScroll}) => {
    const plottedData = twissData[twissAxis.value];
    twissData = plottedData === undefined ? [] : plottedData;

    if (twissData.length !== 0) {
        twissData[0].color = '#CF0000'; // Red
        twissData[1].color = '#0000CF'; // Blue
        twissData[2].color = '#00CF00'; // Green
    }


    // Remove Duplicate z indices in the array
    const removeDuplicateX = (dataArray) => {
      const seen = new Set();
      return dataArray.filter(point => {
        if (seen.has(point.x)) return false;
        seen.add(point.x);
        return true;
      });
    };


    twissData.forEach((entry) => { 
        entry.data = removeDuplicateX(entry.data);
    })

    return <ResponsiveLine
        data={twissData}
        margin={{ top: 10, right: 25, bottom: 40, left: 50 }}
        yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: true, reverse: false }}


        // PLOT BEAM SEGMENTS FOR PREVIEW
        xScale={{
            type: 'linear',
            min: 0,      
            max: totalLen
        }}
        axisBottom={{ legend: 'distance from beam start (m)', legendOffset: 36 }}
        axisLeft={{ legend: twissAxis.value.replace(/[\\$]/g, ''), // Remove \ and $ from the string
                    legendOffset: -40 }}
        colors={(e) => e.color}
        pointSize={5}
        pointColor={{ 'from': 'series.color' }}
        pointBorderWidth={1}
        pointBorderColor={{ from: 'seriesColor' }}
        pointLabelYOffset={-12}
        enableTouchCrosshair={true}
        useMesh={true}
        enableSlices={'x'}
        //onClick={(e) => setZValue(e.points[0].data['x'])} // USE if enableSlices IS 'x'

        onClick={
            scroll
                ? (e) => {
                    setZValue(e.points[0].data['x']) //  USE IF enableSlices IS DISABLED
                    setScroll(false); // Set scroll to false
                }
                : (e) => setZValue(e.points[0].data['x']) //  USE IF enableSlices IS DISABLED
        }
        onMouseMove={
            scroll
                ? (e) => setZValue(e.points[0].data['x'])
                : undefined
        }

        legends={[
            {
                anchor: 'top-left', 
                direction: 'column',
                translateX: 10, 
                itemWidth: 80,
                itemHeight: 22,
                symbolShape: 'circle',
                data: twissData.map((entry) => ({
                    id: entry.id,
                    label: entry.id.replace(/[\\$]/g, ''), // Clean legend item labels
                    fill: entry.color
                }))
            }
        ]}

        //  FOR HOVER SCROLL FUNCTION
        //onMouseMove={(point, event) => {
        //    console.log(point);
        //}}

        layers={[
          // Custom x-axis coloring
          ({ xScale, innerHeight }) => {
        return (
            <>
                {
                    beamline.map((seg, i) => {
                        const params = Object.values(seg)[0];
                        const segment = ( 
                            <line
                              key={i}
                              x1={xScale(params.startPos)}
                              x2={xScale(params.endPos)}
                              y1={innerHeight}
                              y2={innerHeight}
                              stroke={params.color}
                              strokeWidth={10}
                            />
                        );
                        return segment;
                    })
                }

            </>
        )
          },
            'markers', 'axes', 'areas', 'crosshair', 'lines', 'points', 'slices', 'mesh', 'legends'
        ]}
    />
};

export default LineGraph;
