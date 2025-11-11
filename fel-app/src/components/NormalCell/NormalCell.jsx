import { Table } from 'rsuite';
const { Cell } = Table;
import 'rsuite/dist/rsuite.min.css'; 

const NormalCell = ({ rowData, dataKey, ...props }) => {
    return (
        <Cell {...props}>
            {rowData[dataKey]} {/* Display the value of the specified column */}
        </Cell>
    );
};

export default NormalCell;