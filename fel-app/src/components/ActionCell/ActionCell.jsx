import { Table, IconButton } from 'rsuite';
const { Cell } = Table;
import { VscEdit, VscSave, VscRemove } from 'react-icons/vsc';
import 'rsuite/dist/rsuite.min.css'; 


const ActionCell = ({ rowData, dataKey, onEdit, onRemove, ...props }) => {
    return (
    <Cell {...props} style={{ padding: '6px', display: 'flex', gap: '4px' }}>
        <IconButton
        appearance="subtle"
        icon={rowData.status === 'EDIT' ? <VscSave /> : <VscEdit />}
        onClick={() => {
            onEdit(rowData.id);
        }}
        />
        <IconButton
        appearance="subtle"
        icon={<VscRemove />}
        onClick={() => {
            onRemove(rowData.id);
        }}
        />
    </Cell>
    );
};

export default ActionCell;