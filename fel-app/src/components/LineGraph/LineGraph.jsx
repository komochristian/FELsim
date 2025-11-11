import { ResponsiveLine } from '@nivo/line'
import { useState } from 'react';

const LineGraph = ({totalLen, twissData, setZValue, beamline, twissAxis, scroll, setScroll}) => {
    const [unselectedAxis, setUnselectedAxis] = useState([]);

    // Remove Duplicate z indices in the array
    const removeDuplicateX = (dataArray) => {
        const seen = new Set();
        return dataArray.filter(point => {
            if (seen.has(point.x)) return false;
            seen.add(point.x);
            return true;
        });
        };

    const cleanedTwissData = (twissData[twissAxis.value] ?? []).map(entry => ({
        ...entry,
        data: removeDuplicateX(entry.data),
    }));

    if (cleanedTwissData.length !== 0) {
        cleanedTwissData[0].color = !unselectedAxis.includes(cleanedTwissData[0].id) ? '#CF0000' : '#000000'; // Red
        cleanedTwissData[1].color = !unselectedAxis.includes(cleanedTwissData[1].id) ? '#0000CF' : '#000000'; // Blue
        cleanedTwissData[2].color = !unselectedAxis.includes(cleanedTwissData[2].id) ? '#00CF00' : '#000000'; // Green
    }

    // WORK IN PROGRESS TO DISPLAY Z
    // const CustomTooltip = ({ point }) => (
    //     <div
    //         style={{
    //             background: 'white',
    //             padding: '5px',
    //             border: '1px solid #ccc',
    //             borderRadius: '4px',
    //         }}
    //     >
    //         <strong>{point.serieId}</strong>
    //         <br />
    //         x: {point.data.xFormatted}
    //         <br />
    //         y: {point.data.yFormatted}
    //     </div>
    // );

    return <ResponsiveLine
        data={cleanedTwissData.filter(entry => !unselectedAxis.includes(entry.id))}
        margin={{ top: 10, right: 25, bottom: 40, left: 50 }}
        yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false, reverse: false }}


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
        // tooltip={({point}) => <CustomTooltip point={point} />}

        //onClick={(e) => setZValue(e.points[0].data['x'])} // USE if enableSlices IS 'x'

        onClick={
            (e) => {
                setZValue(e.points[0].data['x']);
                if (scroll) setScroll(false);
            }
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
                data: cleanedTwissData.map((entry) => ({
                    id: entry.id,
                    label: entry.id.replace(/[\\$]/g, ''), // Clean legend item labels
                    fill: entry.color
                })),
                onClick: (item) => {
                    if (!unselectedAxis.includes(item.id)) {
                        setUnselectedAxis(prev => [...prev, item.id]);
                    }
                    else {
                        setUnselectedAxis(prev => prev.filter(i => i !== item.id));
                    }

                }
            }
        ]}

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
