import { React, useRef, useEffect, useState } from 'react';
import "../LineGraph/LineGraph.css";
import * as d3 from 'd3';

const LineGraph = ({xAxis, beamSegments}) => {
    const [plot6dValues] = useState([25, 50, 35, 15, 94, 10]);
    const svgRef = useRef();

    useEffect(() => {
        //set up svg
        const w = 400;
        const h = 100;

        
        const svg = d3.select(svgRef.current)
            .attr('width', w)
            .attr('height', h)
            .style('background', '#d3d3d3')
            .style('margin-left', '300')
            .style('margin-top', '50')
            .style('overflow', 'visible');

        svg.selectAll("*").remove(); // Clear previous render

        // set up scaling
        const xScale = d3.scaleLinear()
            .domain([0, plot6dValues.length - 1])
            .range([0, w]);
        const yScale = d3.scaleLinear()
            .domain([0, h])
            .range([h, 0]);
        const generateScaledLine = d3.line()
            .x((d, i) => xScale(i))
            .y(yScale)
            .curve(d3.curveCardinal);

        const xAxis = d3.axisBottom(xScale)
            .ticks(plot6dValues.length)
            .tickFormat(i => i + 1);
        const yAxis = d3.axisLeft(yScale)
            .ticks(5);
        svg.append('g')
            .call(xAxis)
            .attr('transform', `translate(0, ${h})`);
        svg.append('g')
            .call(yAxis);


        // set up axis, data for svg
        svg.selectAll('.line')
            .data([plot6dValues])
            .join('path')
            .attr('d', d => generateScaledLine(d))
            .attr('fill', 'none')
            .attr('stroke', 'black');
    }, [plot6dValues]);

    return (
        <svg ref={svgRef}></svg>
    )
};

export default LineGraph;


