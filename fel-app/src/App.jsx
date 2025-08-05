import React, {useState} from 'react';
import './App.css';
import Dropdown from './components/Dropdown/Dropdown';
import DropdownItem from './components/DropdownItem/DropdownItem';
import BeamSegment from './components/BeamSegment/BeamSegment';
function App()
{
    const [selectedItems, setSelectedItems] = useState([]);
    const items = ['Drift', 'Quadruple Focusing', 'Quadruple Defocusing',
                    'Dipole', 'Dipole wedge'];
    const handleItemClick = (item) => {
        setSelectedItems(prevItems => [...prevItems, item]);
    };
    const handleRename = (index, newName) => {
        setSelectedItems(prev => {
            const updated = [...prev];
            updated[index] = newName;
            return updated;
        });
    };

    const handleDelete = (index) => {
        setSelectedItems(prev => prev.filter((_, i) => i !== index));
    };

    return (
        <>
        <div className="layout">
          <div className="sidebar">
            <h2>Menu</h2>
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
                            name={item}
                            index={index}
                            onRename={handleRename}
                            onDelete={handleDelete}
                        />
                    ))}
                </div>



            {/*<ul>
                {selectedItems.map((item, index) => (
                 <li key={index}>{item}</li>
                 ))}
            </ul>*/}

          </div>
          <div className="main-content">
              <h1>FEL simulation</h1>
              <p>select option on left</p>
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
