import { ResponsiveLine } from '@nivo/line'

const LineGraph = ({twissData, setZValue}) => {
    

    twissData = twissData.slice(21, 24); // TEMPORARY FOR TESTING



    return <ResponsiveLine
        data={twissData}
        margin={{ top: 10, right: 100, bottom: 40, left: 100 }}
        yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: true, reverse: false }}


        // PLOT BEAM SEGMENTS FOR PREVIEW
        //xScale={{
        //    type: 'linear',
        //    min: 0,      
        //    max: 12      
        //}}

        axisBottom={{ legend: 'distance from beam start (m)', legendOffset: 36 }}
        axisLeft={{ legend: 'PLACEHOLDER', legendOffset: -40 }}
        pointSize={10}
        pointColor={{ 'from': 'series.color' }}
        pointBorderWidth={2}
        pointBorderColor={{ from: 'seriesColor' }}
        pointLabelYOffset={-12}
        enableTouchCrosshair={true}
        useMesh={true}
        enableSlices={'x'}
        //onClick={(e) => setZValue(e.data['x'])}  //  USE IF enableSlices IS DISABLED
        onClick={(e) => setZValue(e.points[0].data['x'])} // USE if enableSlices IS 'x'
        legends={[
            {
                anchor: 'top-left',
                direction: 'column',
                translateX: 10,
                itemWidth: 80,
                itemHeight: 22,
                symbolShape: 'circle'
            }
        ]}

        //  FOR HOVER SCROLL FUNCTION
        //onMouseMove={(point, event) => {
        //    console.log(point);
        //}}


        layers={[
        //  // Custom x-axis coloring
        //  ({ xScale, innerHeight }) => {
        //    const segments = [
        //      { from: 'helicopter', to: 'boat', color: 'red' },
        //      { from: 'boat', to: 'train', color: 'green' },
        //    ]

        //    return (
        //      <>
        //        {segments.map((seg, i) => {
        //          const x1 = xScale(seg.from)
        //          const x2 = xScale(seg.to)
        //          return (
        //            <line
        //              key={i}
        //              x1={x1}
        //              x2={x2}
        //              y1={innerHeight}
        //              y2={innerHeight}
        //              stroke={seg.color}
        //              strokeWidth={3}
        //            />
        //          )
        //        })}
        //      </>
        //    )
        //  },
        
            'markers', 'axes', 'areas', 'crosshair', 'lines', 'points', 'slices', 'mesh', 'legends'
        ]}
        
        

    />
};

export default LineGraph;
