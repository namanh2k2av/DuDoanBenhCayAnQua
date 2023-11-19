const dropArea = document.querySelector('.drag-area');
const dragText = dropArea.querySelector('header');
const button = dropArea.querySelector('button');
const input = dropArea.querySelector('input');

button.addEventListener('click', () =>{
    input.click();
})

input.addEventListener('change', () =>{
    const file = this.files[0];
    showFile(file);
})

function showFile(file){
    let fileType = file.type;
    let validExtensions = ['image/jpeg', 'image/jpg', 'image/png'];
    if(validExtensions.includes(fileType)){
        let fileReader = new FileReader();

        fileReader.onload = () =>{
            let fileUrl = fileReader.result;
            let imgTag = `<img src="${fileUrl}">`
            dropArea.innerHTML = imgTag

        }
        fileReader.readAsDataURL(file);
    }

    else{
        alert("Đây không phải file ảnh");
        dragText.textContent = "Kéo và thả để tải file lên"
    }
}