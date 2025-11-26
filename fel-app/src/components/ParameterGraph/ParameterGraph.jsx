import { ResponsiveLine } from "@nivo/line";
import { useState } from "react";

const ParameterGraph = ({data, parameter_name, twiss_target}) => {
    const [unselectedAxis, setUnselectedAxis] = useState([]);

    function convertToNivoLineData(inputArray) {
        if (!inputArray || inputArray.length === 0) {
            return [];
        }
        const series = {
            x: [],
            y: [],
            z: []
        };
    
        inputArray.forEach(entry => {
            const { parameter_value, data } = entry;
            const plotData = data.filter(d => d.twiss_parameter === twiss_target)[0];

            series.x.push({ x: parameter_value, y: plotData.x });
            series.y.push({ x: parameter_value, y: plotData.y });
            series.z.push({ x: parameter_value, y: plotData.z });
            // console.log('series:', series);
        });
    

        return [
            { id: `${twiss_target}: x`, data: series.x, color: '#CF0000' },
            { id: `${twiss_target}: y`, data: series.y, color: '#0000CF' },
            { id: `${twiss_target}: z`, data: series.z, color: '#00CF00' },
        ];
    }

    function handleNivoColor(data) {
        if (data.length !== 0) {
            data[0].color = !unselectedAxis.includes(data[0].id) ? '#CF0000' : '#000000'; // Red
            data[1].color = !unselectedAxis.includes(data[1].id) ? '#0000CF' : '#000000'; // Blue
            data[2].color = !unselectedAxis.includes(data[2].id) ? '#00CF00' : '#000000'; // Green
        }
    }

    const nivoData = convertToNivoLineData(data);
    handleNivoColor(nivoData);

    return (
        <ResponsiveLine
            data={nivoData.filter(entry => !unselectedAxis.includes(entry.id))}
            margin={{ top: 10, right: 25, bottom: 40, left: 50 }}
            xScale={{ type: 'linear', min: 'auto', max: 'auto' }}
            yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: false, reverse: false }}
            axisBottom={{ legend: `${parameter_name} value`, legendOffset: 36 }}
            axisLeft={{ legend: twiss_target, legendOffset: -40 }}
            colors={((e) => e.color)} // Red, Green, Blue
            pointSize={6}
            pointColor={{ theme: 'background' }}
            pointBorderWidth={2}
            pointBorderColor={{ from: 'serieColor' }}
            pointLabelYOffset={-12}
            useMesh={true}
            legends={[
                {
                    anchor: 'top-left', 
                    direction: 'column',
                    translateX: 10, 
                    itemWidth: 80,
                    itemHeight: 22,
                    symbolShape: 'circle',
                    data: nivoData.map((entry) => ({
                        id: entry.id,
                        label: entry.id, 
                        fill: entry.color,
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
        />
    );

};

export default ParameterGraph;