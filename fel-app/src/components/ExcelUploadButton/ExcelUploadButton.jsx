import { React, useState } from 'react';
import * as XLSX from 'xlsx';

const ExcelUploadButton = ({excelToAPI}) => {
    
    const fileHandler = (e) => {
        const file = e.target.files[0];
        const reader = new FileReader();

        
        reader.onload = (event) => { 
          const workbook = XLSX.read(event.target.result, { type: 'binary' });
          const sheetName = workbook.SheetNames[0];
          const sheet = workbook.Sheets[sheetName];
          const sheetData = XLSX.utils.sheet_to_json(sheet);

          excelToAPI(sheetData);
        };

        reader.readAsBinaryString(file);
    };

    return <div>
                <input type="file" accept=".xlsx" onChange={fileHandler} />
            </div>
};

export default ExcelUploadButton;
