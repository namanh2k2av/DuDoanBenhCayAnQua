import React, { useState } from 'react';
import './ABC.css';

function NhanDien() {

    const [image, setImage] = useState();

    const chosseFile = (e) =>{
        const file = e.target.files[0];

        file.preview = URL.createObjectURL(file);

        setImage(file);
    }

    return(
        <>
            <h1 style={{textAlign: "center"}}>Chuẩn đoán bệnh dựa trên lá cho cây ăn quả</h1>

            <form className='mainForm'>
                <p>
                    <label><b>Chọn ảnh lá cây bị bệnh cần chuẩn đoán</b></label><br />
                    <input type="file" name="" id="imageFile" onChange={chosseFile}/>
                </p>

                <p>
                    {
                        image && (
                            <img src={image.preview} alt="" id='image'/>
                        )
                    }
                </p>

                <p>
                    <label><b>Dự đoán bệnh</b></label><br />
                    <textarea name="" id="" cols="40" rows="8"></textarea>
                </p>

                <button className='button'>
                    Kiểm tra
                </button>
            </form>
        </>
    )
}

export default NhanDien;