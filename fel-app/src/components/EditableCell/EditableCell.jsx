
import { Table, Input, InputNumber, Whisper, Tooltip } from 'rsuite';
const { Cell } = Table;
import 'rsuite/dist/rsuite.min.css'; 

const EditableCell = ({ rowData, dataType, dataKey, onChange, onEdit, ...props }) => {
    const fieldMap = {
        string: Input,
        number: (props) => <InputNumber step={0.01} {...props} />,
    };

    const editing = rowData.status === 'EDIT';
  
    const Field = fieldMap[dataType];
    let value = rowData[dataKey];

    const parseInput = (input, dataType) => {
        switch (dataType) {
            case 'number':
              return Number(input); // Convert string to number
            case 'string':
              return String(input); // Ensure it's a string
            default:
              return input; // Return as is for unrecognized types
          } 
    }
  
    return (
      <Whisper
      trigger="hover"
      speaker={<Tooltip>click to select insert position</Tooltip>}
      placement="top" // Adjust placement as needed (e.g., 'bottom', 'left', 'right')
    >
      <Cell
        {...props}
        className={editing ? 'table-cell-editing' : ''}
      >
        {editing ? (
          <Field
            defaultValue={value}
            onChange={value => {
              onChange?.(rowData.id, dataKey, parseInput(value, dataType), editing);
            }}
            onClick={event => event.stopPropagation()}
          />
        ) : (
          value
        )}
      </Cell>
      </Whisper>

    );
  };

  export default EditableCell;