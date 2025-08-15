import { ResponsiveLine } from '@nivo/line'

const LineGraph = () => {
    
    const data =
        [
  {
    "id": "japan",
    "data": [
      {
        "x": "plane",
        "y": 46
      },
      {
        "x": "helicopter",
        "y": 147
      },
      {
        "x": "boat",
        "y": 269
      },
      {
        "x": "train",
        "y": 150
      },
      {
        "x": "subway",
        "y": 258
      },
      {
        "x": "bus",
        "y": 0
      },
      {
        "x": "car",
        "y": 263
      },
      {
        "x": "moto",
        "y": 187
      },
      {
        "x": "bicycle",
        "y": 117
      },
      {
        "x": "horse",
        "y": 49
      },
      {
        "x": "skateboard",
        "y": 219
      },
      {
        "x": "others",
        "y": 0
      }
    ]
  },
  {
    "id": "france",
    "data": [
      {
        "x": "plane",
        "y": 251
      },
      {
        "x": "helicopter",
        "y": 37
      },
      {
        "x": "boat",
        "y": 7
      },
      {
        "x": "train",
        "y": 91
      },
      {
        "x": "subway",
        "y": 143
      },
      {
        "x": "bus",
        "y": 128
      },
      {
        "x": "car",
        "y": 9
      },
      {
        "x": "moto",
        "y": 91
      },
      {
        "x": "bicycle",
        "y": 246
      },
      {
        "x": "horse",
        "y": 32
      },
      {
        "x": "skateboard",
        "y": 187
      },
      {
        "x": "others",
        "y": 275
      }
    ]
  },
  {
    "id": "us",
    "data": [
      {
        "x": "plane",
        "y": 246
      },
      {
        "x": "helicopter",
        "y": 239
      },
      {
        "x": "boat",
        "y": 284
      },
      {
        "x": "train",
        "y": 200
      },
      {
        "x": "subway",
        "y": 179
      },
      {
        "x": "bus",
        "y": 276
      },
      {
        "x": "car",
        "y": 3
      },
      {
        "x": "moto",
        "y": 271
      },
      {
        "x": "bicycle",
        "y": 33
      },
      {
        "x": "horse",
        "y": 79
      },
      {
        "x": "skateboard",
        "y": 111
      },
      {
        "x": "others",
        "y": 202
      }
    ]
  },
  {
    "id": "germany",
    "data": [
      {
        "x": "plane",
        "y": 136
      },
      {
        "x": "helicopter",
        "y": 4
      },
      {
        "x": "boat",
        "y": 126
      },
      {
        "x": "train",
        "y": 81
      },
      {
        "x": "subway",
        "y": 171
      },
      {
        "x": "bus",
        "y": 215
      },
      {
        "x": "car",
        "y": 64
      },
      {
        "x": "moto",
        "y": 202
      },
      {
        "x": "bicycle",
        "y": 8
      },
      {
        "x": "horse",
        "y": 138
      },
      {
        "x": "skateboard",
        "y": 14
      },
      {
        "x": "others",
        "y": 115
      }
    ]
  },
  {
    "id": "norway",
    "data": [
      {
        "x": "plane",
        "y": 128
      },
      {
        "x": "helicopter",
        "y": 104
      },
      {
        "x": "boat",
        "y": 121
      },
      {
        "x": "train",
        "y": 34
      },
      {
        "x": "subway",
        "y": 152
      },
      {
        "x": "bus",
        "y": 243
      },
      {
        "x": "car",
        "y": 186
      },
      {
        "x": "moto",
        "y": 288
      },
      {
        "x": "bicycle",
        "y": 147
      },
      {
        "x": "horse",
        "y": 270
      },
      {
        "x": "skateboard",
        "y": 39
      },
      {
        "x": "others",
        "y": 298
      }
    ]
  }
];
    return <ResponsiveLine /* or Line for fixed dimensions */
        data={data}
        margin={{ top: 10, right: 100, bottom: 40, left: 100 }}
        yScale={{ type: 'linear', min: 'auto', max: 'auto', stacked: true, reverse: false }}
        axisBottom={{ legend: 'transportation', legendOffset: 36 }}
        axisLeft={{ legend: 'count', legendOffset: -40 }}
        pointSize={10}
        pointColor={{ theme: 'background' }}
        pointBorderWidth={2}
        pointBorderColor={{ from: 'seriesColor' }}
        pointLabelYOffset={-12}
        enableTouchCrosshair={true}
        useMesh={true}
        legends={[
            {
                anchor: 'bottom-right',
                direction: 'column',
                translateX: 100,
                itemWidth: 80,
                itemHeight: 22,
                symbolShape: 'circle'
            }
        ]}
        onClick={(e) => console.log(e)}
        layers={[
          // Custom x-axis coloring
          ({ xScale, innerHeight }) => {
            const segments = [
              { from: 'helicopter', to: 'boat', color: 'red' },
              { from: 'boat', to: 'train', color: 'green' },
            ]

            return (
              <>
                {segments.map((seg, i) => {
                  const x1 = xScale(seg.from)
                  const x2 = xScale(seg.to)
                  return (
                    <line
                      key={i}
                      x1={x1}
                      x2={x2}
                      y1={innerHeight}
                      y2={innerHeight}
                      stroke={seg.color}
                      strokeWidth={3}
                    />
                  )
                })}
              </>
            )
          },
          // Default layers
          'grid', 'markers', 'areas', 'lines', 'slices', 'points', 'axes', 'legends',
        ]}
    />
};

export default LineGraph;
