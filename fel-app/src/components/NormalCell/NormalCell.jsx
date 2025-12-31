import { Table, Whisper, Tooltip } from 'rsuite';
const { Cell } = Table;
import 'rsuite/dist/rsuite.min.css'; 

const NormalCell = ({ rowData, dataKey, ...props }) => {
    return (
        <Cell {...props}>
          <Whisper
            trigger="hover"
            speaker={<Tooltip>click to select insert position</Tooltip>}
            placement="top" // Adjust placement as needed (e.g., 'bottom', 'left', 'right')
          >
            <a>{rowData[dataKey]}</a>
          </Whisper>
        </Cell>
    );
};

export default NormalCell;