/* global $ */

function xjson(xjson) {
    var obj = JSON.parse(xjson);
    for (var i = obj.length - 1; i >= 0; i--) {
        console.log(obj[i]);
    }
}

class Main {

    constructor() {
        this.canvas = document.getElementById('main');
        this.input = document.getElementById('input');
        this.real_width = 40;
        this.real_height = 30;
        this.patch_size = 10;
        this.canvas.width  = this.patch_size * this.real_width + 1; // 16 * 28 + 1
        this.canvas.height = this.patch_size * this.real_height + 1; // 16 * 28 + 1
        this.ctx = this.canvas.getContext('2d');
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mouseup',   this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.initialize();
    }
    initialize() {
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.lineWidth = 1;
        this.ctx.strokeRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.lineWidth = 0.05;
        for (var i = 0; i < this.real_width; i++) {
            this.ctx.beginPath();
            this.ctx.moveTo((i + 1) * this.patch_size,   0);
            this.ctx.lineTo((i + 1) * this.patch_size, this.canvas.height);
            this.ctx.closePath();
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(  0, (i + 1) *  this.patch_size);
            this.ctx.lineTo(this.canvas.width, (i + 1) *  this.patch_size);
            this.ctx.closePath();
            this.ctx.stroke();
        }
        this.drawInput();
        $('#output td').text('').removeClass('success');
    }
    onMouseDown(e) {
        this.canvas.style.cursor = 'default';
        this.drawing = true;
        this.prev = this.getPosition(e.clientX, e.clientY);
    }
    onMouseUp() {
        this.drawing = false;
        this.drawInput();
    }
    onMouseMove(e) {
        if (this.drawing) {
            var curr = this.getPosition(e.clientX, e.clientY);
            this.ctx.lineWidth = 16;
            this.ctx.lineCap = 'round';
            this.ctx.beginPath();
            this.ctx.moveTo(this.prev.x, this.prev.y);
            this.ctx.lineTo(curr.x, curr.y);
            this.ctx.stroke();
            this.ctx.closePath();
            this.prev = curr;
        }
    }
    getPosition(clientX, clientY) {
        var rect = this.canvas.getBoundingClientRect();
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }
    drawInput() {
        var ctx = this.input.getContext('2d');
        var img = new Image();
        img.onload = () => {
            var inputs = [];
            var small = document.createElement('canvas').getContext('2d');
            small.drawImage(img, 0, 0, img.width, img.height, 0, 0, this.real_width, this.real_height);
            var data = small.getImageData(0, 0, this.real_width, this.real_height).data;
            for (var i = 0; i < this.real_height; i++) {
                for (var j = 0; j < this.real_width; j++) {
                    var n = 4 * (i * this.real_width + j);
                    inputs[i * this.real_width + j] = (data[n + 0] + data[n + 1] + data[n + 2]) / 3;
                    ctx.fillStyle = 'rgb(' + [data[n + 0], data[n + 1], data[n + 2]].join(',') + ')';
                    ctx.fillRect(j * 4, i * 4, 4, 4);
                }
            }
            if (Math.min(...inputs) === 255) {
                return;
            }
            //console.log(JSON.stringify(inputs));
            $.post("/api",
                {
                    data:JSON.stringify(inputs),
                },
                function(data,status){
                    //alert("Data: " + data + "\nStatus: " + status);
                    xjson(data);
                    var obj = JSON.parse(data);
                    //for (var i = 0; i < 3; i++) {
                    //    $('#output tr').eq(i+1).find('td').eq(1).text(obj[i+3]+"  (Predict Rate:"+obj[i]+")");
                    //}
                    //for (var i = 0; i < 3; i++) {
                        //$('#output tr').eq(i+1).find('td').eq(0).text(obj[i+3+3]+"  (概率:"+obj[i+3]+")");
                    //}
            });
        };
        img.src = this.canvas.toDataURL();
    }
}

$(() => {
    var main = new Main();
    $('#clear').click(() => {
        main.initialize();
    });
});
