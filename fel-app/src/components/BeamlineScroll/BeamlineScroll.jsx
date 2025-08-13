import {React} from 'react'
import '../BeamlineScroll/BeamlineScroll.css'

const BeamlineScroll = ({intervalList, changeIntervalList}) => {
    
   return <div className='beamlineScrollbar'>
            {intervalList.map((num) => (
                 <div className='scrollItem' key={num}>
                        {num}
                        </div>
                ))}
          </div>
};

export default BeamlineScroll;
