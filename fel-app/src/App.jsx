import React, {useState, useEffect} from 'react';
import './App.css';
import Dropdown from './components/Dropdown/Dropdown';
import DropdownItem from './components/DropdownItem/DropdownItem';
import BeamSegment from './components/BeamSegment/BeamSegment';
function App()
{
    const [beamSegmentInfo, setData] = useState(null);

    useEffect(() => {
        fetch('http://127.0.0.1:8000/get-beamsegmentinfo')
            .then((response) => response.json())
            .then((json) => setData(json))
            .catch((err) => console.error("Error loading beam segment info:", err));
        }, []);


    const [selectedItems, setSelectedItems] = useState([]);
    useEffect(() => {

    console.log("Updated selectedItems:", selectedItems);
}, [selectedItems]);

    if (!beamSegmentInfo) return <div>Loading...</div>;
    const items = Object.keys(beamSegmentInfo);

    const handleItemClick = (item) => {
        setSelectedItems(prevItems => [...prevItems, {[item]: beamSegmentInfo[item]}]);
    };

    const handleDelete = (index) => {
        setSelectedItems(prev => prev.filter((_, i) => i !== index));
    };



//    const handleParamChange = (index, key, newValue) => {
//        
//        setSelectedItems(prev =>
//            prev.map((item, i) =>
//                if (i != index) return item;
//
//                i === index
//                    ? { ...item, [key]: parseFloat(newValue) }
//                    : item
//            )
//        );
//    };
    const handleParamChange = (index, paramKey, newValue) => {
        setSelectedItems(prev =>
            prev.map((item, i) => {
                if (i !== index) return item;
                console.log('handleparams index:', index);
                const topKey = Object.keys(item)[0];
                const updatedParams = {
                    ...item[topKey],
                    [paramKey]: parseFloat(newValue)
                };
    
                return {
                    [topKey]: updatedParams
                };
            })
        );
    };

    return (
        <>
        <div className="layout">
          <div className="sidebar">
            <h2>FEL simulator</h2>
             <Dropdown buttonText="Add Segment" 
                     contentText={
                             <>
                                 {items.map((item) => (
                                     <DropdownItem key={item}
                                                   onClick={() => handleItemClick(item)}
                                     >
                                         {`${item}`}
                                     </DropdownItem>
                                 ))}
                             </>
                     }
             />
            <h3>Beam setup</h3>
               <div className="scrollBox">
                    {selectedItems.map((item, index) => (
                        <BeamSegment 
                            key={index}
                            name={Object.keys(item)[0]}
                            params={item}
                            index={index}
                            onDelete={handleDelete}
                            onChanges={handleParamChange}
                        />
                    ))}
                </div>
          </div>
          <div className="main-content">
          </div>
          <div className="linegraph ">
            <h1>graph here</h1>
          </div>
          <div className="twiss">
            <h6>Twiss options</h6>
          </div>
        </div>
        </>
    );
}

export default App;
